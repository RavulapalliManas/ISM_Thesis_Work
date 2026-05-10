import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

BASE = Path(__file__).resolve().parents[1]
PYTHON = BASE / ".brain" / "Scripts" / "python.exe"
RUNNER = BASE / "analysis" / "run_speedhd_ablation.py"
ABLATION_DIR = BASE / "results" / "ablation"
LAUNCH_LOG = ABLATION_DIR / "parallel_launcher.jsonl"


def log(event, payload):
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **payload}
    with open(LAUNCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    try:
        from analysis.ablation_dashboard import tick

        tick(show_terminal=False)
    except Exception:
        pass


def gpu_memory():
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode != 0:
            return None
        used, total, util, temp = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return {"used": used, "total": total, "free": total - used, "util": util, "temp": temp}
    except Exception:
        return None


def build_jobs(max_steps, checkpoint_steps):
    jobs = []
    for mode in ["full", "ablated", "degraded"]:
        for seed in [0, 1]:
            jobs.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "cmd": [
                        str(PYTHON),
                        str(RUNNER),
                        "--hd-mode",
                        mode,
                        "--seed",
                        str(seed),
                        "--max-steps",
                        str(max_steps),
                        "--checkpoint-steps",
                        checkpoint_steps,
                    ],
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40_000)
    parser.add_argument("--checkpoint-steps", default="10000,20000,30000,40000")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--min-free-vram-mb", type=float, default=700.0)
    parser.add_argument("--max-ram-percent", type=float, default=88.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.max_steps, args.checkpoint_steps)
    log(
        "launcher_start",
        {
            "max_parallel": args.max_parallel,
            "max_steps": args.max_steps,
            "batch": args.batch,
            "anchors": args.anchors,
            "jobs": [{"mode": j["mode"], "seed": j["seed"]} for j in jobs],
        },
    )

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))
        log("launcher_dry_run_complete", {})
        return

    env_base = os.environ.copy()
    env_base["PRNN_ABLATION_BATCH"] = str(args.batch)
    env_base["PRNN_ABLATION_ANCHORS"] = str(args.anchors)
    env_base["PRNN_ABLATION_PROGRESS_INTERVAL"] = str(args.progress_interval)

    pending = list(jobs)
    active = []
    completed = []
    failed = []

    while pending or active:
        still_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_active.append(item)
            else:
                item["returncode"] = ret
                completed.append(item)
                if ret != 0:
                    failed.append(item)
                log("job_complete", {"mode": item["mode"], "seed": item["seed"], "returncode": ret})
        active = still_active

        mem = gpu_memory()
        ram = psutil.virtual_memory()
        can_launch = (
            len(active) < args.max_parallel
            and pending
            and (mem is None or mem["free"] >= args.min_free_vram_mb)
            and ram.percent < args.max_ram_percent
        )
        while can_launch:
            job = pending.pop(0)
            run_dir = ABLATION_DIR / f"hd_{job['mode']}" / f"seed_{job['seed']:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "process_stdout.log"
            stdout = open(stdout_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(BASE),
                env=env_base,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({**job, "proc": proc, "stdout": stdout, "stdout_path": str(stdout_path)})
            log(
                "job_start",
                {
                    "mode": job["mode"],
                    "seed": job["seed"],
                    "pid": proc.pid,
                    "active": len(active),
                    "pending": len(pending),
                    "gpu": mem,
                    "ram_percent": ram.percent,
                },
            )
            mem = gpu_memory()
            ram = psutil.virtual_memory()
            can_launch = (
                len(active) < args.max_parallel
                and pending
                and (mem is None or mem["free"] >= args.min_free_vram_mb)
                and ram.percent < args.max_ram_percent
            )

        log(
            "launcher_tick",
            {
                "active": [{"mode": x["mode"], "seed": x["seed"], "pid": x["proc"].pid} for x in active],
                "pending": len(pending),
                "completed": len(completed),
                "failed": len(failed),
                "gpu": mem,
                "ram_percent": ram.percent,
            },
        )
        time.sleep(30)

    for item in completed:
        try:
            item["stdout"].close()
        except Exception:
            pass
    log("launcher_complete", {"completed": len(completed), "failed": len(failed)})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
