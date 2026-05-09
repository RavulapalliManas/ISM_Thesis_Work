#!/usr/bin/env python3
"""
=============================================================================
  GPU Experiment Manager — pRNN Computational Neuroscience Research
=============================================================================
  Single-file orchestrator: auto-detects hardware, maximises GPU utilisation,
  guides experimental design, and provides a live training dashboard.

  Requirements (auto-installed on first run):
      pip install rich psutil

  Usage:
      python gpu_manager.py                          # full interactive mode
      python gpu_manager.py --dir ./project5         # point at experiment dir
      python gpu_manager.py --script train.py        # direct script path
      python gpu_manager.py --config run.json        # load saved config
      python gpu_manager.py --dry-run                # preview without launching
=============================================================================
"""
import os, sys, json, time, subprocess, threading, re, shutil, signal
import argparse, copy, textwrap
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import multiprocessing as mp

def _ensure(pkgs: Dict[str, str]):
    for import_name, pip_name in pkgs.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"[setup] Installing {pip_name}...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name, "-q"], check=True)

_ensure({"rich": "rich", "psutil": "psutil"})

import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn)
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.text import Text
from rich import box
from rich.align import Align
from rich.rule import Rule
from rich.columns import Columns
from rich.padding import Padding

console = Console()

try:
    import torch
    _TORCH = torch.cuda.is_available()
except ImportError:
    _TORCH = False

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareProfile:
    def __init__(self):
        self.gpu_name = "CPU-only (no GPU detected)"
        self.num_gpus = 0
        self.vram_total_mb = 0
        self.cpu_cores = mp.cpu_count()
        self.ram_total_gb = psutil.virtual_memory().total / 1e9
        self._detect_gpu()
        psutil.cpu_percent(interval=None)

    def _detect_gpu(self):
        if _TORCH:
            try:
                props = torch.cuda.get_device_properties(0)
                self.gpu_name = props.name
                self.vram_total_mb = props.total_memory // (1024 ** 2)
                self.num_gpus = torch.cuda.device_count()
                return
            except Exception:
                pass
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,count", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                self.gpu_name = parts[0].strip()
                self.vram_total_mb = int(parts[1].strip())
                self.num_gpus = int(parts[2].strip()) if len(parts) > 2 else 1
        except Exception:
            pass

    def gpu_stats(self) -> Tuple[float, int, int]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                p = r.stdout.strip().split(",")
                return float(p[0]), int(p[1]), int(p[2])
        except Exception:
            pass
        if _TORCH:
            try:
                free, total = torch.cuda.mem_get_info(0)
                used = (total - free) // (1024 ** 2)
                return 0.0, used, free // (1024 ** 2)
            except Exception:
                pass
        return 0.0, 0, self.vram_total_mb

    def max_parallel(self, vram_per_seed_mb: int = 1300) -> int:
        if self.vram_total_mb == 0:
            return max(1, self.cpu_cores // 4)
        DRIVER_RESERVE = 600
        avail = self.vram_total_mb - DRIVER_RESERVE
        gpu_cap = max(1, int(avail / (vram_per_seed_mb * 1.15)))
        cpu_cap = max(1, self.cpu_cores // 2)
        ram_avail = psutil.virtual_memory().available / 1e9
        ram_cap = max(1, int(ram_avail / 5))
        return min(gpu_cap, cpu_cap, ram_cap)

    def probe_vram(self, cmd_prefix: List[str], probe_seconds: int = 40) -> Optional[int]:
        _, baseline, _ = self.gpu_stats()
        probe_cmd = cmd_prefix + ["--seed", "999", "--n_epochs", "5"]
        try:
            proc = subprocess.Popen(probe_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            peak = baseline
            for _ in range(probe_seconds * 2):
                time.sleep(0.5)
                _, used, _ = self.gpu_stats()
                peak = max(peak, used)
                if proc.poll() is not None:
                    break
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                pass
            delta = peak - baseline
            return max(800, delta + 200) if delta > 100 else None
        except Exception:
            return None

    def summary_table(self) -> Table:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("k", style="bold cyan")
        t.add_column("v")
        t.add_row("GPU", self.gpu_name)
        t.add_row("VRAM", f"{self.vram_total_mb:,} MB" if self.vram_total_mb else "N/A")
        t.add_row("CPU cores", str(self.cpu_cores))
        t.add_row("System RAM", f"{self.ram_total_gb:.1f} GB")
        return t

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCIENTIFIC ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════

_KB = {
    "stat_power": {
        "rationale": (
            "For Mann-Whitney U (nonparametric, appropriate for small-N RNN experiments) "
            "5 seeds per condition achieves ~60-70% power for medium effects (Cohen d ~ 0.8). "
            "8 seeds reaches the standard 80% threshold. "
            "For Kruskal-Wallis across >= 3 conditions, 5 seeds/condition is the hard minimum."
        ),
        "per_n": {
            1: ("🔴", "DEBUGGING ONLY — no statistical claims possible."),
            2: ("🔴", "Report range only. No inferential statistics."),
            3: ("🟡", "Minimum variance estimate. Effect sizes highly uncertain."),
            4: ("🟡", "Marginal. Mann-Whitney possible but ~50% power."),
            5: ("🟢", "Publication minimum. ~60-70% power for medium effects."),
            8: ("✅", "Recommended. ~80% power. Suitable for primary claims."),
            10: ("💪", "Strong. >85% power. Use for high-stakes comparisons."),
        },
    },
    "hyperparams": {
        "n_steps": {
            "default": 80000,
            "note": "Training steps (not epochs). 80k matches project5 ExperimentConfig default.",
            "advice": "80k for all primary experiments. 10 for smoke tests.",
            "warn_low": 20000,
        },
        "hidden_size": {
            "default": 500,
            "note": "500 matches Levenstein et al. and all configs in this codebase.",
            "advice": "Keep 500 for baseline replication. Only change for scaling experiments.",
        },
        "lr": {
            "default": 3e-3,
            "note": "trainNet.py default is 3e-3 (relative LR). Project5 uses RMSProp.",
            "advice": "3e-3 for legacy path. Project5 training loop has its own LR schedule.",
        },
        "batch_size": {
            "default": 8,
            "note": "B=8 is the ExperimentConfig default for project5_symmetry.",
            "advice": "B=8 for project5. B=1 only for strict Levenstein replication via trainNet.py.",
        },
        "T": {
            "default": 200,
            "note": "Sequence length for BPTT. T=200 is validated default across all configs.",
            "advice": "T=200 for all primary experiments. T=50/T=600 only in Phase 4b sweep.",
        },
        "k": {
            "default": 5,
            "note": "Rollout steps in pRNN theta architecture. k=5 matches Levenstein.",
            "advice": "Keep k=5. This is the biologically grounded theta parameter.",
        },
        "arena_size": {
            "default": 18,
            "note": "18x18 grid is the standard. 12/24/30 only in Phase 1 arena scaling.",
            "advice": "18x18 for all primary experiments.",
        },
        "F": {
            "default": 7,
            "note": "Visual field edge length. F=7 is standard, F=3/5 in Phase 2b.",
            "advice": "F=7 for all primary experiments.",
        },
        "U": {
            "default": 3,
            "note": "Landmark colour classes. U=3 is standard, U=0-4 in Phase 2a sweep.",
            "advice": "U=3 for standard runs.",
        },
        "symmetry": {
            "default": "S1",
            "note": "Landmark symmetry: S4 (C4), S2 (C2), S1 (asymmetric). ODI held constant.",
            "advice": "Run S4, S2, S1 in parallel across your seed sweep.",
        },
    },
    "templates": {
        "P0_baseline": {
            "desc": "Phase 0 baseline gate — L-shape arena, must reach sRSA_euclid > 0.40.",
            "min_seeds": 5, "rec_seeds": 9,
            "srsa_target": "> 0.40 (gate criterion)",
            "goal": "Confirm sRSA_euclid > 0.40 at convergence before any manipulation.",
            "params": {"hidden_size": 500, "T": 200, "k": 5, "n_steps": 80000, "batch_size": 8},
        },
        "symmetry_sweep": {
            "desc": "S4/S2/S1 landmark symmetry comparison (Project 5 ISM, run_symmetry_sweep.py).",
            "min_seeds": 5, "rec_seeds": 9,
            "goal": "Demonstrate monotonic relationship between symmetry and degeneracy.",
            "metrics": ["sRSA_euclid", "sRSA_city", "SCI", "DTG", "manifold_id"],
            "params": {"hidden_size": 500, "T": 200, "k": 5, "n_steps": 80000},
        },
        "phase1_arena_scaling": {
            "desc": "Phase 1 — arena size sweep (12/18/24/30) + L-shape control.",
            "min_seeds": 5, "rec_seeds": 9,
            "goal": "Characterise sRSA dependence on arena scale.",
            "params": {"hidden_size": 500, "T": 200, "k": 5, "n_steps": 80000},
        },
        "phase2a_landmark_density": {
            "desc": "Phase 2a — landmark density U=0..4 on 18x18 square.",
            "min_seeds": 5, "rec_seeds": 9,
            "goal": "Find optimal landmark density for spatial learning.",
            "params": {"hidden_size": 500, "T": 200, "k": 5, "n_steps": 80000},
        },
        "hd_ablation": {
            "desc": "HD ablation study (S4 arena, 3 conditions × 2 seeds, 40k steps each).",
            "min_seeds": 2, "rec_seeds": 2,
            "goal": "Test necessity of HD for global orientational stability in C4-symmetric arena.",
            "metrics": ["sRSA", "PAA_gain", "RA", "decode_err"],
            "params": {"n_steps": 40000, "hidden_size": 500, "T": 200, "k": 5}
        },
    },
}

class ScientificAdvisor:
    def seed_status(self, n: int) -> Tuple[str, str]:
        notes = _KB["stat_power"]["per_n"]
        key = min(notes.keys(), key=lambda k: (abs(k - n), -k))
        return notes[key]

    def param_advice(self, name: str) -> Optional[str]:
        return _KB["hyperparams"].get(name, {}).get("advice")

    def param_note(self, name: str) -> Optional[str]:
        return _KB["hyperparams"].get(name, {}).get("note")

    def param_default(self, name: str) -> Any:
        return _KB["hyperparams"].get(name, {}).get("default")

    def param_warn_low(self, name: str, value) -> Optional[str]:
        wl = _KB["hyperparams"].get(name, {}).get("warn_low")
        if wl is not None and isinstance(value, (int, float)) and value < wl:
            return f"Warning: {name}={value} is below recommended minimum ({wl})."
        return None

    def templates(self) -> Dict:
        return _KB["templates"]

    def power_table(self) -> Table:
        t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
        t.add_column("Seeds", width=7, justify="center")
        t.add_column("Assessment")
        for n in sorted(_KB["stat_power"]["per_n"]):
            icon, txt = _KB["stat_power"]["per_n"][n]
            t.add_row(str(n), f"{icon}  {txt}")
        return t

    def dashboard_tip(self, gpu_util, n_active, n_done, n_total, srsas):
        text = Text()
        text.append("🧠  Advisor\n", style="bold green")
        if n_active > 0:
            if gpu_util < 50:
                text.append("⚠  GPU < 50% — consider more parallel seeds.\n", style="yellow")
            elif gpu_util > 96:
                text.append("⚠  GPU near capacity — reduce if OOM.\n", style="red")
            else:
                text.append(f"✓  GPU healthy at {gpu_util:.0f}%\n", style="green")
        text.append("\n")
        if srsas:
            gate = sum(1 for s in srsas if s > 0.40)
            text.append(f"sRSA gate (>0.40): {gate}/{len(srsas)}\n",
                        style="green" if gate == len(srsas) else "yellow")
            text.append(f"Mean sRSA: {sum(srsas)/len(srsas):.3f}\n")
        else:
            text.append("sRSA gate: awaiting results…\n", style="dim")
        text.append("\n")
        pct = 100 * n_done / max(n_total, 1)
        text.append(f"Progress: {n_done}/{n_total} ({pct:.0f}%)\n")
        text.append("\nCtrl-C to stop all runs.\n", style="dim")
        return text

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS: Dict[str, Any] = {
    "hidden_size": 500, "lr": 3e-3, "batch_size": 8, "T": 200,
    "k": 5, "n_steps": 80000, "arena_size": 18, "F": 7, "U": 3,
    "symmetry": "S1", "output_dir": "./gpu_manager_outputs",
}

class ExperimentConfig:
    def __init__(self, script, params=None, seeds=None, name=""):
        self.script = str(Path(script).resolve())
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seeds = seeds or [0]
        self.name = name or Path(script).stem

    def build_cmd(self, seed: int) -> List[str]:
        cmd = [sys.executable, self.script]
        for k, v in self.params.items():
            if k != "output_dir":
                cmd += [f"--{k}", str(v)]
        cmd += ["--seed", str(seed)]
        return cmd

    def save(self, path: str):
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"script": self.script, "params": self.params,
                       "seeds": self.seeds, "name": self.name}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(d["script"], d.get("params"), d.get("seeds"), d.get("name", ""))


def discover_scripts(base: str) -> List[Dict]:
    base = Path(base)
    patterns = ["train*.py", "run*.py", "main*.py", "experiment*.py",
                "*pRNN*.py", "*rnn*.py", "*sweep*.py"]
    found = set()
    for p in patterns:
        found.update(base.rglob(p))
    results = []
    for s in sorted(found):
        if any(x in str(s) for x in ["__pycache__", ".git", "site-packages"]):
            continue
        results.append({"path": str(s), "name": s.name, "dir": str(s.parent),
                        "rel": str(s.relative_to(base))})
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROCESS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class SeedRun:
    _RE_STEP_TOT = re.compile(r"(?:epoch|step|iter(?:ation)?)\s+(\d+)\s*/\s*(\d+)", re.I)
    _RE_STEP_ONLY = re.compile(r"(?:epoch|step|iter(?:ation)?)\s*[=:]\s*(\d+)", re.I)
    _RE_LOSS = re.compile(r"(?:train_?)?loss\s*[=:]\s*([\d.eE+\-]+)", re.I)
    _RE_SRSA = re.compile(r"s(?:patial_?)?rsa\s*[=:]\s*([\d.]+)", re.I)
    _RE_PAA = re.compile(r"paa_gain\s*[=:]\s*([\d.eE+\-]+)", re.I)

    def __init__(self, seed, cfg, run_id):
        self.seed = seed
        self.cfg = cfg
        self.run_id = run_id
        self.proc = None
        self.t_start = None
        self.t_end = None
        self.status = "queued"
        self.progress = 0.0
        self.step = 0
        self.total = cfg.params.get("n_steps", 80000)
        self.loss = None
        self.srsa = None
        self.log_path = None
        self._lock = threading.Lock()

    @property
    def elapsed(self):
        if self.t_start is None: return 0.0
        return (self.t_end or time.time()) - self.t_start

    @property
    def eta(self):
        if self.progress <= 0.01 or self.progress >= 1: return None
        return self.elapsed * (1 - self.progress) / self.progress

    def launch(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, f"seed_{self.seed:02d}.log")
        cmd = self.cfg.build_cmd(self.seed)
        lf = open(self.log_path, "w")
        lf.write(f"# Command: {' '.join(cmd)}\n# Started: {datetime.now()}\n\n")
        lf.flush()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        self.t_start = time.time()
        self.status = "running"
        threading.Thread(target=self._reader, args=(lf,), daemon=True).start()

    def _reader(self, lf):
        for line in self.proc.stdout:
            lf.write(line); lf.flush()
            self._parse(line)
        ret = self.proc.wait()
        self.t_end = time.time()
        self.status = "done" if ret == 0 else "failed"
        if ret == 0: self.progress = 1.0
        lf.write(f"\n# Finished ({self.status}) in {timedelta(seconds=int(self.elapsed))}\n")
        lf.close()

    def _parse(self, line):
        m = self._RE_STEP_TOT.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            self.step = cur; self.total = tot
            self.progress = min(1.0, cur / max(tot, 1))
        else:
            m = self._RE_STEP_ONLY.search(line)
            if m:
                self.step = int(m.group(1))
                self.progress = min(1.0, self.step / max(self.total, 1))
        m = self._RE_LOSS.search(line)
        if m:
            try: self.loss = float(m.group(1))
            except ValueError: pass
        m = self._RE_SRSA.search(line)
        if m:
            try: self.srsa = float(m.group(1))
            except ValueError: pass
        m = self._RE_PAA.search(line)
        if m:
            if not hasattr(self, 'metrics'):
                self.metrics = {}
            try: self.metrics["PAA_gain"] = float(m.group(1))
            except ValueError: pass

    def kill(self):
        if self.proc and self.status == "running":
            try: self.proc.terminate()
            except Exception: pass
            self.status = "failed"


class ExperimentQueue:
    def __init__(self, runs, max_parallel, out_dir):
        self.all_runs = runs
        self.max_parallel = max_parallel
        self.out_dir = out_dir
        self._queue = list(runs)
        self.active = []
        self.completed = []
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        with self._lock:
            for r in self.active: r.kill()

    def _loop(self):
        while self._running:
            with self._lock:
                done = [r for r in self.active if r.status in ("done", "failed")]
                for r in done:
                    self.active.remove(r); self.completed.append(r)
                while len(self.active) < self.max_parallel and self._queue:
                    run = self._queue.pop(0)
                    rdir = os.path.join(self.out_dir, f"run_{run.run_id:03d}_seed{run.seed}")
                    run.launch(rdir); self.active.append(run)
            if not self._queue and not self.active:
                break
            time.sleep(1.5)

    @property
    def n_total(self): return len(self.all_runs)
    @property
    def n_done(self): return len(self.completed)
    @property
    def is_complete(self):
        return len(self.completed) == self.n_total and not self._queue

    def overall_progress(self):
        if not self.all_runs: return 0.0
        return (sum(1.0 for r in self.completed) + sum(r.progress for r in self.active)) / self.n_total

    def est_wall_time(self):
        finished = [r for r in self.completed if r.t_end]
        if finished:
            avg = sum(r.elapsed for r in finished) / len(finished)
        else:
            active_ripe = [r for r in self.active if r.progress > 0.05]
            if not active_ripe: return None
            avg = sum(r.elapsed / r.progress for r in active_ripe) / len(active_ripe)
        return avg * max(1, self.n_total / self.max_parallel)

    def completed_srsas(self):
        return [r.srsa for r in self.completed if r.srsa is not None]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

_STATUS_COLOR = {"queued": "dim white", "running": "cyan", "done": "green", "failed": "red"}
_STATUS_ICON = {"queued": "○", "running": "⟳", "done": "✓", "failed": "✗"}

def _bar(frac, width=28, color="green"):
    filled = max(0, min(width, int(frac * width)))
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="dim")
    return t

class Dashboard:
    def __init__(self, hw, queue, advisor, cfg):
        self.hw = hw; self.queue = queue; self.advisor = advisor
        self.cfg = cfg; self.t0 = time.time()
        self._last_gpu = (0.0, 0, hw.vram_total_mb)

    def _refresh_gpu(self):
        self._last_gpu = self.hw.gpu_stats()

    def _header(self):
        elapsed = str(timedelta(seconds=int(time.time() - self.t0)))
        overall = self.queue.overall_progress()
        est = self.queue.est_wall_time()
        remain = str(timedelta(seconds=int(max(0, est - (time.time()-self.t0))))) if est else "—"
        t = Text(justify="center")
        t.append("⚡ GPU Experiment Manager", style="bold white"); t.append("  │  ")
        t.append(f"Elapsed: {elapsed}", style="yellow"); t.append("  │  ")
        t.append(f"Remaining: ~{remain}", style="cyan"); t.append("  │  ")
        t.append(f"{overall*100:.1f}%", style="bold green" if overall > 0.8 else "bold cyan")
        t.append(f"  │  {self.queue.n_done}/{self.queue.n_total} done")
        return Panel(Align.center(t), style="blue", padding=(0, 1))

    def _gpu_panel(self):
        util, used, free = self._last_gpu
        vram_pct = used / max(self.hw.vram_total_mb, 1)
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        tbl = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        tbl.add_column("Label", style="cyan", width=12)
        tbl.add_column("Bar", ratio=3)
        tbl.add_column("Value", width=18, justify="right")
        gc = "green" if util > 75 else ("yellow" if util > 40 else "red")
        tbl.add_row("GPU Util", _bar(util/100, color=gc), f"[{gc}]{util:.0f}%[/{gc}]")
        vc = "red" if vram_pct > 0.88 else ("yellow" if vram_pct > 0.70 else "cyan")
        tbl.add_row("VRAM", _bar(vram_pct, color=vc), f"[{vc}]{used:,}/{self.hw.vram_total_mb:,} MB[/{vc}]")
        tbl.add_row("CPU", _bar(cpu_pct/100, color="blue"), f"{cpu_pct:.0f}% ({self.hw.cpu_cores} cores)")
        tbl.add_row("RAM", _bar(ram.percent/100, color="magenta"), f"{ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB")
        n_active = len(self.queue.active)
        return Panel(tbl, title=f"[bold]{self.hw.gpu_name}[/bold] — [cyan]{n_active}/{self.queue.max_parallel} slots[/cyan]",
                     border_style="blue", padding=(0, 1))

    def _runs_panel(self):
        tbl = Table(box=box.SIMPLE_HEAD, expand=True, show_header=True, padding=(0,1))
        tbl.add_column("ID", width=5); tbl.add_column("Seed", width=6)
        tbl.add_column("State", width=9); tbl.add_column("Progress", ratio=2)
        tbl.add_column("Step", width=12, justify="right"); tbl.add_column("Loss", width=9, justify="right")
        tbl.add_column("sRSA", width=7, justify="right"); tbl.add_column("ETA", width=11, justify="right")
        show = self.queue.active[:8] + self.queue.completed[-4:] + self.queue._queue[:2]
        for r in show:
            c = _STATUS_COLOR[r.status]; ic = _STATUS_ICON[r.status]
            step_s = f"{r.step}/{r.total}" if r.total else "—"
            loss_s = f"{r.loss:.4f}" if r.loss is not None else "—"
            srsa_s = f"[{'green bold' if r.srsa > 0.40 else 'yellow'}]{r.srsa:.3f}[/]" if r.srsa is not None else "—"
            if r.eta is not None: eta_s = str(timedelta(seconds=int(r.eta)))
            elif r.status == "done": eta_s = f"[dim]({str(timedelta(seconds=int(r.elapsed)))})[/dim]"
            else: eta_s = "—"
            tbl.add_row(f"{r.run_id:03d}", f"[{c}]{ic}[/{c}] {r.seed}", f"[{c}]{r.status}[/{c}]",
                        _bar(r.progress, width=22, color=c), step_s, f"[yellow]{loss_s}[/yellow]", srsa_s, eta_s)
        return Panel(tbl, title=f"[bold]Seed Runs[/bold] [{len(self.queue._queue)} queued]",
                     border_style="cyan", padding=(0, 1))

    def _stats_panel(self):
        t = Text(); elapsed = time.time() - self.t0; est = self.queue.est_wall_time()
        t.append("━━ Timing ━━\n", style="bold yellow")
        t.append(f"Wall time: {str(timedelta(seconds=int(elapsed)))}\n")
        if est:
            t.append(f"Est. remaining: {str(timedelta(seconds=int(max(0, est - elapsed))))}\n")
        t.append("\n━━ Summary ━━\n", style="bold cyan")
        n_ok = sum(1 for r in self.queue.completed if r.status == "done")
        n_fail = sum(1 for r in self.queue.completed if r.status == "failed")
        t.append(f"✓ Done: {n_ok}\n", style="green")
        t.append(f"⟳ Active: {len(self.queue.active)}\n", style="cyan")
        t.append(f"○ Queued: {len(self.queue._queue)}\n", style="dim")
        if n_fail: t.append(f"✗ Failed: {n_fail}\n", style="red bold")
        srsas = self.queue.completed_srsas()
        if srsas:
            t.append("\n━━ sRSA ━━\n", style="bold magenta")
            t.append(f"Mean: {sum(srsas)/len(srsas):.3f}\n")
            gate = sum(1 for s in srsas if s > 0.40)
            t.append(f"Gate (>0.40): {gate}/{len(srsas)}\n", style="green" if gate == len(srsas) else "yellow")
        return Panel(t, title="[bold]Statistics[/bold]", border_style="yellow", padding=(1, 1))

    def _advisor_panel(self):
        util = self._last_gpu[0]
        tip = self.advisor.dashboard_tip(util, len(self.queue.active), self.queue.n_done,
                                          self.queue.n_total, self.queue.completed_srsas())
        return Panel(tip, title="[bold green]Advisor[/bold green]", border_style="green", padding=(1, 1))

    def _footer(self):
        t = Text(justify="center")
        t.append("Ctrl-C", style="bold yellow"); t.append(" graceful stop  │  logs in: ", style="dim")
        t.append(self.queue.out_dir, style="cyan")
        return Panel(t, style="dim", padding=(0, 1))

    def render(self):
        self._refresh_gpu()
        L = Layout()
        L.split_column(Layout(self._header(), name="header", size=4), Layout(name="body"),
                       Layout(self._footer(), name="footer", size=3))
        L["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=1))
        L["left"].split_column(Layout(self._gpu_panel(), name="gpu", size=8),
                               Layout(self._runs_panel(), name="runs"))
        L["right"].split_column(Layout(self._stats_panel(), name="stats", size=14),
                                Layout(self._advisor_panel(), name="advisor"))
        return L

    def run(self):
        with Live(self.render(), refresh_per_second=2, screen=True) as live:
            try:
                while not self.queue.is_complete:
                    live.update(self.render()); time.sleep(0.5)
                live.update(self.render()); time.sleep(3)
            except KeyboardInterrupt:
                pass
        console.print("\n[yellow]Stopping all active runs…[/yellow]")
        self.queue.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTERACTIVE SETUP WIZARDS
# ═══════════════════════════════════════════════════════════════════════════════

def wizard_select_script(base_dir=None):
    console.print(Rule("[bold cyan]Step 1 — Select Training Script[/bold cyan]"))
    if base_dir is None:
        base_dir = Prompt.ask("[cyan]Base directory to scan[/cyan]", default=str(Path.cwd()))
    base_dir = str(Path(base_dir).resolve())
    console.print(f"[dim]Scanning {base_dir}…[/dim]")
    found = discover_scripts(base_dir)
    if not found:
        console.print("[yellow]No training scripts found automatically.[/yellow]")
        return Prompt.ask("[cyan]Path to training script[/cyan]")
    tbl = Table(box=box.SIMPLE_HEAD, show_header=True)
    tbl.add_column("#", width=4, style="dim"); tbl.add_column("Script", style="cyan")
    tbl.add_column("Relative path")
    for i, e in enumerate(found):
        tbl.add_row(str(i+1), e["name"], e["rel"])
    console.print(tbl)
    n = IntPrompt.ask("[cyan]Select number[/cyan]", default=1)
    return found[max(0, min(n-1, len(found)-1))]["path"]

def wizard_template(advisor):
    console.print(Rule("[bold cyan]Step 2 — Experiment Template (optional)[/bold cyan]"))
    tmpls = advisor.templates()
    tbl = Table(box=box.ROUNDED, show_header=True, expand=True)
    tbl.add_column("#", width=3); tbl.add_column("Template", style="cyan", width=22)
    tbl.add_column("Description", ratio=2); tbl.add_column("Seeds", width=8, justify="center")
    names = list(tmpls.keys())
    for i, k in enumerate(names):
        v = tmpls[k]
        tbl.add_row(str(i+1), k, v["desc"], f"[yellow]{v['rec_seeds']}[/yellow]")
    console.print(tbl)
    if Confirm.ask("[cyan]Load a template?[/cyan]", default=False):
        n = IntPrompt.ask("Template number", default=1)
        return tmpls[names[max(0, min(n-1, len(names)-1))]]
    return None

def wizard_hyperparams(base, advisor):
    console.print(Rule("[bold cyan]Step 3 — Hyperparameters[/bold cyan]"))
    console.print("[dim]Press Enter to accept default.[/dim]\n")
    params = dict(base)
    edit_order = [
        ("n_steps", "Training steps", int), ("hidden_size", "Hidden size", int),
        ("lr", "Learning rate", float), ("batch_size", "Batch size", int),
        ("T", "Sequence length T (BPTT)", int), ("k", "Rollout steps k", int),
        ("arena_size", "Arena size N", int), ("F", "Visual field edge F", int),
        ("U", "Landmark classes U", int), ("symmetry", "Symmetry (S1/S2/S4)", str),
        ("output_dir", "Output directory", str),
    ]
    for name, label, dtype in edit_order:
        current = params.get(name, DEFAULT_PARAMS.get(name, ""))
        note = advisor.param_note(name)
        if note: console.print(f"  [dim italic]{note[:110]}[/dim italic]")
        raw = Prompt.ask(f"  [cyan]{label}[/cyan]", default=str(current))
        try: val = dtype(raw)
        except ValueError: val = current
        warn = advisor.param_warn_low(name, val)
        if warn: console.print(f"  [bold red]{warn}[/bold red]")
        params[name] = val; console.print()
    return params

def wizard_seeds_parallel(advisor, hw, vram_per_seed):
    console.print(Rule("[bold cyan]Step 4 — Seeds & Parallelism[/bold cyan]"))
    console.print(advisor.power_table()); console.print()
    hw_max = hw.max_parallel(vram_per_seed)
    console.print(f"[cyan]Hardware estimate:[/cyan] up to [bold]{hw_max}[/bold] parallel seeds")
    console.print(f"  GPU: {hw.gpu_name}  ({hw.vram_total_mb:,} MB VRAM)\n")
    n_seeds = IntPrompt.ask("[cyan]Total seeds[/cyan]", default=9)
    seeds = list(range(n_seeds))
    rec = max(1, min(hw_max, n_seeds, hw_max - 1 if hw_max > 1 else 1))
    n_par = IntPrompt.ask(f"[cyan]Max parallel[/cyan] (hw max={hw_max})", default=rec)
    n_par = max(1, min(n_par, hw_max, n_seeds))
    icon, txt = advisor.seed_status(n_seeds)
    console.print(f"\n[bold]Assessment:[/bold]  {icon}  {txt}\n")
    return seeds, n_par

def wizard_vram(hw, script, params, probe):
    HEURISTIC = 1300
    if hw.vram_total_mb == 0: return HEURISTIC
    if probe:
        console.print("[cyan]Probing VRAM…[/cyan]")
        cmd_prefix = [sys.executable, script]
        for k, v in params.items():
            if k not in ("output_dir", "n_steps"):
                cmd_prefix += [f"--{k}", str(v)]
        probed = hw.probe_vram(cmd_prefix, probe_seconds=45)
        if probed:
            console.print(f"[green]Probed VRAM/seed:[/green] {probed:,} MB"); return probed
        console.print("[yellow]Probe failed — using heuristic.[/yellow]")
    if Confirm.ask(f"[cyan]Know VRAM per run? (heuristic = {HEURISTIC} MB)[/cyan]", default=False):
        return IntPrompt.ask("[cyan]VRAM per seed (MB)[/cyan]", default=HEURISTIC)
    return HEURISTIC

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_summary(queue, t0, out_dir):
    console.print(Rule("[bold green]Experiment Complete[/bold green]"))
    n_ok = sum(1 for r in queue.completed if r.status == "done")
    n_fail = sum(1 for r in queue.completed if r.status == "failed")
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column("k", style="cyan"); tbl.add_column("v")
    tbl.add_row("Completed", f"[green]{n_ok}[/green] / {queue.n_total}")
    if n_fail: tbl.add_row("Failed", f"[red]{n_fail}[/red]")
    tbl.add_row("Wall time", str(timedelta(seconds=int(time.time() - t0))))
    tbl.add_row("Log dir", out_dir)
    srsas = queue.completed_srsas()
    if srsas:
        gate = sum(1 for s in srsas if s > 0.40)
        tbl.add_row("sRSA mean", f"{sum(srsas)/len(srsas):.3f} (gate >0.40: {gate}/{len(srsas)})")
    console.print(tbl)

def main():
    ap = argparse.ArgumentParser(description="GPU Experiment Manager — pRNN Research")
    ap.add_argument("--dir", default=None, help="Base directory to scan")
    ap.add_argument("--script", default=None, help="Direct path to training script")
    ap.add_argument("--config", default=None, help="JSON config from a previous run")
    ap.add_argument("--output", default="./gpu_manager_outputs", help="Output directory")
    ap.add_argument("--seeds", type=int, default=None, help="# seeds (skips prompt)")
    ap.add_argument("--parallel", type=int, default=None, help="Max parallel (skips prompt)")
    ap.add_argument("--probe-vram", action="store_true", help="Auto-probe VRAM")
    ap.add_argument("--dry-run", action="store_true", help="Configure only")
    args = ap.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold white]⚡  GPU Experiment Manager[/bold white]\n"
        "[dim]  pRNN Research Dashboard — Hippocampal Computational Neuroscience  [/dim]",
        border_style="blue", padding=(1, 6)))
    console.print()

    console.print("[cyan]Detecting hardware…[/cyan]")
    hw = HardwareProfile()
    console.print(hw.summary_table())
    util, used_mb, free_mb = hw.gpu_stats()
    console.print(f"[dim]Current GPU: {util:.0f}% util │ {free_mb:,} MB VRAM free[/dim]\n")

    advisor = ScientificAdvisor()

    if args.config:
        cfg = ExperimentConfig.load(args.config)
        console.print(f"[green]Loaded config:[/green] {args.config}")
        script = cfg.script; params = cfg.params; seeds = cfg.seeds
        n_par = args.parallel or hw.max_parallel()
    else:
        tmpl = wizard_template(advisor)
        params = {**DEFAULT_PARAMS, **(tmpl["params"] if tmpl else {})}
        script = args.script or wizard_select_script(args.dir)
        if not Path(script).exists():
            console.print(f"[red]Script not found: {script}[/red]"); sys.exit(1)
        params = wizard_hyperparams(params, advisor)
        vram = wizard_vram(hw, script, params, args.probe_vram)
        if args.seeds and args.parallel:
            seeds = list(range(args.seeds)); n_par = args.parallel
        else:
            seeds, n_par = wizard_seeds_parallel(advisor, hw, vram)

    out_dir = args.output or params.get("output_dir", "./gpu_manager_outputs")

    console.print(Rule("[bold]Launch Configuration[/bold]"))
    cfg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    cfg_table.add_column("Parameter", style="cyan"); cfg_table.add_column("Value")
    cfg_table.add_row("Script", script)
    cfg_table.add_row("Seeds", f"{len(seeds)} ({seeds[0]}–{seeds[-1]})")
    cfg_table.add_row("Parallel", f"{n_par} (GPU max ≈ {hw.max_parallel()})")
    cfg_table.add_row("Output dir", out_dir)
    for k in ["n_steps", "hidden_size", "batch_size", "T", "k", "lr"]:
        if k in params: cfg_table.add_row(k, str(params[k]))
    console.print(cfg_table)

    icon, seed_txt = advisor.seed_status(len(seeds))
    console.print(Panel(f"{icon}  [bold]{seed_txt}[/bold]",
                        title="[bold green]🧠  Seed Assessment[/bold green]",
                        border_style="green", padding=(1, 2)))

    if args.dry_run:
        console.print("[yellow]--dry-run: not launching.[/yellow]"); return
    if not Confirm.ask("\n[bold green]Launch experiments?[/bold green]", default=True):
        console.print("[yellow]Aborted.[/yellow]"); return

    os.makedirs(out_dir, exist_ok=True)
    cfg_obj = ExperimentConfig(script, params, seeds)
    cfg_path = os.path.join(out_dir, "experiment_config.json")
    cfg_obj.save(cfg_path)
    console.print(f"[dim]Config saved → {cfg_path}[/dim]\n")

    runs = [SeedRun(s, cfg_obj, i) for i, s in enumerate(seeds)]
    queue = ExperimentQueue(runs, max_parallel=n_par, out_dir=out_dir)

    t0 = time.time()
    console.print(f"[bold green]Launching {len(seeds)} seeds ({n_par} parallel)…[/bold green]")
    queue.start(); time.sleep(2)

    dash = Dashboard(hw, queue, advisor, cfg_obj)
    dash.run()
    print_final_summary(queue, t0, out_dir)

if __name__ == "__main__":
    main()
