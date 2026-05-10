import argparse
import json
import subprocess
import time
from html import escape
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
ABLATION_DIR = BASE / "results" / "ablation"
HTML_PATH = ABLATION_DIR / "dashboard.html"
STATUS_PATH = ABLATION_DIR / "dashboard_status.json"


def read_events(path):
    events = []
    if not path.exists():
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event": "parse_error", "raw": line})
    return events


def latest_run_state(run_dir):
    events = read_events(run_dir / "dashboard_events.jsonl")
    state = {
        "run": str(run_dir.relative_to(ABLATION_DIR)),
        "status": "not_started",
        "step": 0,
        "total": None,
        "loss": None,
        "gpu_mb": None,
        "gpu_reserved_mb": None,
        "spatial_sRSA": None,
        "last_event": None,
        "last_time": None,
        "flags": [],
        "checkpoints": [],
        "hd_check": None,
        "training_check": None,
    }
    for ev in events:
        state["last_event"] = ev.get("event")
        state["last_time"] = ev.get("time")
        event = ev.get("event")
        if event == "run_start":
            state["status"] = "running"
            state["total"] = ev.get("max_steps")
        elif event == "hd_check":
            state["hd_check"] = ev
            state["gpu_mb"] = ev.get("gpu_mb")
            state["gpu_reserved_mb"] = ev.get("gpu_reserved_mb")
        elif event == "training_check":
            state["training_check"] = ev
            state["step"] = max(state["step"], int(ev.get("step", 0)))
            state["loss"] = ev.get("loss")
        elif event == "progress":
            state["status"] = "running"
            state["step"] = int(ev.get("step", state["step"]))
            state["total"] = ev.get("total", state["total"])
            state["loss"] = ev.get("loss", state["loss"])
            state["gpu_mb"] = ev.get("gpu_mb", state["gpu_mb"])
            state["gpu_reserved_mb"] = ev.get("gpu_reserved_mb", state["gpu_reserved_mb"])
        elif event == "checkpoint":
            state["step"] = int(ev.get("step", state["step"]))
            state["loss"] = ev.get("loss", state["loss"])
            state["spatial_sRSA"] = ev.get("spatial_sRSA", state["spatial_sRSA"])
            state["checkpoints"].append(ev)
        elif event == "flag":
            state["flags"].append(ev)
            state["status"] = "flagged"
        elif event == "run_complete":
            state["status"] = "complete"
            state["loss"] = ev.get("loss", state["loss"])
            if state["total"] is not None:
                state["step"] = max(state["step"], int(state["total"]))
    return state


def collect_states():
    run_dirs = []
    for mode in ["hd_full", "hd_ablated", "hd_degraded"]:
        for seed in ["seed_00", "seed_01"]:
            run_dirs.append(ABLATION_DIR / mode / seed)
    return [latest_run_state(p) for p in run_dirs]


def gpu_status():
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode != 0:
            return {"raw": res.stderr.strip(), "available": False}
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "available": True,
            "name": parts[0],
            "util_percent": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
            "temperature_c": float(parts[4]),
            "power_w": parts[5],
        }
    except Exception as exc:
        return {"available": False, "raw": f"{type(exc).__name__}: {exc}"}


def write_status(states, gpu):
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu,
        "runs": states,
    }
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def pct(state):
    total = state.get("total")
    if not total:
        return 0.0
    return min(100.0, 100.0 * float(state.get("step", 0)) / float(total))


def fmt(x, digits=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_html(payload):
    gpu = payload["gpu"]
    rows = []
    for st in payload["runs"]:
        flags = "<br>".join(escape(f.get("message", "flag")) for f in st["flags"]) or "-"
        hd = st.get("hd_check") or {}
        train = st.get("training_check") or {}
        rows.append(
            "<tr>"
            f"<td>{escape(st['run'])}</td>"
            f"<td class='{escape(st['status'])}'>{escape(st['status'])}</td>"
            f"<td>{st['step']}/{st.get('total') or '-'}</td>"
            f"<td><div class='bar'><span style='width:{pct(st):.1f}%'></span></div>{pct(st):.1f}%</td>"
            f"<td>{fmt(st.get('loss'))}</td>"
            f"<td>{fmt(st.get('spatial_sRSA'), 3)}</td>"
            f"<td>{fmt(st.get('gpu_mb'), 1)} / {fmt(st.get('gpu_reserved_mb'), 1)}</td>"
            f"<td>{fmt(hd.get('heading_sum_mean'), 3)}</td>"
            f"<td>{fmt(hd.get('speed_mean'), 3)}</td>"
            f"<td>{fmt(train.get('loss_delta'), 4)}</td>"
            f"<td>{len(st['checkpoints'])}</td>"
            f"<td>{escape(st.get('last_time') or '-')}</td>"
            f"<td>{flags}</td>"
            "</tr>"
        )
    if gpu.get("available"):
        gpu_line = (
            f"{escape(gpu['name'])} | util {gpu['util_percent']:.0f}% | "
            f"VRAM {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB | "
            f"{gpu['temperature_c']:.0f} C | {escape(str(gpu['power_w']))} W"
        )
    else:
        gpu_line = escape(gpu.get("raw", "GPU unavailable"))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>SpeedHD Ablation Dashboard</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 24px; background: #f7f3ea; color: #1f2520; }}
    h1 {{ margin-bottom: 4px; }}
    .card {{ background: #fffaf0; border: 1px solid #d8cdb8; border-radius: 12px; padding: 14px 16px; margin: 14px 0; box-shadow: 0 1px 8px rgba(0,0,0,0.05); }}
    table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e7dece; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #273326; color: white; position: sticky; top: 0; }}
    .running {{ color: #2166AC; font-weight: bold; }}
    .complete {{ color: #1B7837; font-weight: bold; }}
    .flagged {{ color: #B2182B; font-weight: bold; }}
    .not_started {{ color: #777; }}
    .bar {{ display: inline-block; width: 120px; height: 9px; background: #e4ded2; border-radius: 8px; overflow: hidden; margin-right: 8px; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, #2166AC, #67A9CF); }}
    code {{ background: #eee4d1; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>SpeedHD Ablation Dashboard</h1>
  <div>Updated: {escape(payload['updated_at'])}. Auto-refreshes every 10 seconds.</div>
  <div class="card"><strong>GPU:</strong> {gpu_line}</div>
  <div class="card"><strong>Encoding:</strong> SpeedHD 5D <code>[speed, hd_0, hd_1, hd_2, hd_3]</code>. Ablated/degraded runs preserve speed and alter heading dims 1:5 only.</div>
  <table>
    <thead>
      <tr>
        <th>Run</th><th>Status</th><th>Step</th><th>Progress</th><th>Loss</th><th>sRSA</th>
        <th>GPU MB alloc/res</th><th>HD sum</th><th>Speed mean</th><th>Loss delta @500</th>
        <th>Checkpoints</th><th>Last update</th><th>Flags</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


def print_terminal(payload):
    print("\033[2J\033[H", end="")
    print(f"SpeedHD Ablation Dashboard | updated {payload['updated_at']}")
    gpu = payload["gpu"]
    if gpu.get("available"):
        print(
            f"GPU: {gpu['name']} | util {gpu['util_percent']:.0f}% | "
            f"VRAM {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB | "
            f"{gpu['temperature_c']:.0f} C"
        )
    else:
        print(f"GPU: {gpu.get('raw', 'unavailable')}")
    print("-" * 120)
    print(f"{'run':24} {'status':12} {'step':>12} {'pct':>7} {'loss':>10} {'sRSA':>7} {'gpuMB':>13} {'flags'}")
    for st in payload["runs"]:
        flags = ",".join(f.get("message", "flag") for f in st["flags"]) or "-"
        print(
            f"{st['run']:24} {st['status']:12} {str(st['step']) + '/' + str(st.get('total') or '-'):>12} "
            f"{pct(st):6.1f}% {fmt(st.get('loss')):>10} {fmt(st.get('spatial_sRSA'), 3):>7} "
            f"{fmt(st.get('gpu_mb'), 1) + '/' + fmt(st.get('gpu_reserved_mb'), 1):>13} {flags}"
        )
    print("-" * 120)
    print(f"HTML: {HTML_PATH}")


def tick(show_terminal=True):
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    states = collect_states()
    gpu = gpu_status()
    payload = write_status(states, gpu)
    render_html(payload)
    if show_terminal:
        print_terminal(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Refresh continuously.")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--no-terminal", action="store_true")
    args = parser.parse_args()
    if args.watch:
        while True:
            tick(show_terminal=not args.no_terminal)
            time.sleep(args.interval)
    else:
        tick(show_terminal=not args.no_terminal)


if __name__ == "__main__":
    main()
