"""Turn bench_matmul.py CSVs into Markdown tables, crossover points and charts.

  uv run report.py --latest        # newest CSV per (tag, device)
  uv run report.py results/*.csv   # explicit files

Writes results/summary.md, results/gflops.png and results/latency.png.
Paste summary.md sections into docs/cpu-vs-gpu.md sections 4 and 5.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open() as f:
            for r in csv.DictReader(f):
                r["n"] = int(r["n"])
                r["median_s"] = float(r["median_s"])
                r["gflops"] = float(r["gflops"])
                rows.append(r)
    return rows


def latest_files() -> list[Path]:
    newest: dict[tuple[str, str], Path] = {}
    for p in sorted(RESULTS.glob("*.csv")):  # names end in a timestamp, so sorted = oldest first
        tag, device = p.stem.split("_")[:2]
        if tag == "smoke":
            continue
        newest[(tag, device)] = p
    return list(newest.values())


def series_key(r: dict) -> str:
    return f"{r['tag']}/{r['device']}/{r['dtype']}/{r['mode']}"


def table(rows: list[dict], value: str, fmt) -> str:
    by = defaultdict(dict)
    for r in rows:
        by[series_key(r)][r["n"]] = r[value]
    keys = sorted(by)
    ns = sorted({r["n"] for r in rows})
    out = ["| N | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)]
    for n in ns:
        out.append(f"| {n} | " + " | ".join(fmt(by[k][n]) if n in by[k] else "" for k in keys) + " |")
    return "\n".join(out)


def crossovers(rows: list[dict]) -> list[str]:
    """Smallest N from which the accelerator stays faster than the CPU (same tag if possible)."""
    cpu = defaultdict(dict)
    for r in rows:
        if r["device"] == "cpu" and r["dtype"] == "fp32":
            cpu[r["tag"]][r["n"]] = r["median_s"]
    if not cpu:
        return ["No CPU fp32 baseline found: run the CPU sweep too."]
    fallback = next(iter(cpu.values()))
    acc = defaultdict(dict)
    for r in rows:
        if r["device"] != "cpu" and r["mode"] in ("compute", "e2e"):
            acc[(r["tag"], r["device"], r["dtype"], r["mode"])][r["n"]] = r["median_s"]
    lines = []
    for (tag, dev, dt, mode), pts in sorted(acc.items()):
        base = cpu.get(tag, fallback)
        common = sorted(set(pts) & set(base))
        wins = [n for n in common if pts[n] < base[n]]
        # crossover = first N after which every larger shared N is also a win
        cross = next((n for n in common if all(m in wins for m in common if m >= n)), None)
        cpu_wins = [n for n in common if n not in wins]
        lines.append(
            f"- **{tag}/{dev} {dt} {mode}** vs CPU fp32: GPU wins from **N = {cross if cross else 'never (in range)'}**"
            + (f"; CPU wins at N = {', '.join(map(str, cpu_wins))}" if cpu_wins else "")
        )
    return lines


def naive_inflation(rows: list[dict]) -> str:
    comp = {(r["tag"], r["device"], r["dtype"], r["n"]): r for r in rows if r["mode"] == "compute"}
    out = ["| series | N | honest GFLOP/s | naive GFLOP/s | inflation | naive % of peak |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["mode"] != "naive":
            continue
        c = comp.get((r["tag"], r["device"], r["dtype"], r["n"]))
        if not c:
            continue
        out.append(f"| {r['tag']}/{r['device']}/{r['dtype']} | {r['n']} | {c['gflops']:,.0f} | {r['gflops']:,.0f} | "
                   f"{r['gflops'] / c['gflops']:.1f}x | {r['pct_peak']} |")
    return "\n".join(out) if len(out) > 2 else "_No naive-mode rows._"


def charts(rows: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["(matplotlib not installed: charts skipped)"]
    made = []
    for value, fname, ylabel in (("gflops", "gflops.png", "GFLOP/s"), ("median_s", "latency.png", "seconds per matmul")):
        by = defaultdict(list)
        for r in rows:
            if r["mode"] != "naive":
                by[series_key(r)].append((r["n"], r[value]))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for k, pts in sorted(by.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=k)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (N x N matmul)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=120)
        plt.close(fig)
        made.append(str(RESULTS / fname))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()
    files = latest_files() if args.latest or not args.files else args.files
    if not files:
        raise SystemExit("no CSVs found in results/")
    rows = load(files)
    RESULTS.mkdir(exist_ok=True)

    md = [
        "# Matmul sweep summary",
        "",
        "Sources: " + ", ".join(f"`{p.name}`" for p in files),
        "",
        "## Achieved GFLOP/s (section 4.1)",
        table(rows, "gflops", lambda v: f"{v:,.1f}"),
        "",
        "## Median latency, ms (section 4.2)",
        table(rows, "median_s", lambda v: f"{v * 1e3:.4f}"),
        "",
        "## Crossover (section 5.1)",
        *crossovers(rows),
        "",
        "## Naive timing inflation (section 5.4)",
        naive_inflation(rows),
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    for c in charts(rows):
        print("chart:", c)


if __name__ == "__main__":
    main()
