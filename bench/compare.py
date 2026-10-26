"""Stock OpenCV vs COOL on the same machine, from bench.py outputs (interleaved runs: stock, COOL, COOL, stock).

    python -m bench.compare --stock results/cool/bench_stock_a.json results/cool/bench_stock_b.json \
                            --cool results/cool/bench_cool_a.json results/cool/bench_cool_b.json

Each stage's time is the median over its runs; speed-up = stock / COOL (> 1 means COOL is faster).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def load(paths):
    t, prov = defaultdict(list), None
    for p in paths:
        d = json.load(open(p))
        prov = prov or d["provenance"]
        for r in d["stages"]:
            if "case" in r:
                t[(r["case"], r["stage"])].append(r["seconds"])
    return {k: statistics.median(v) for k, v in t.items()}, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", nargs="+", required=True)
    ap.add_argument("--cool", nargs="+", required=True)
    a = ap.parse_args()
    s, ps = load(a.stock)
    c, pc = load(a.cool)
    print(f"stock: OpenCV {ps['opencv']} ({ps.get('instance_type', ps['machine'])}, {ps['threads']} threads)  "
          f"COOL: OpenCV {pc['opencv']} (cool={pc['cool']})\n")
    print("| Case | Stage | Stock (s) | COOL (s) | Speed-up |")
    print("|---|---|---|---|---|")
    by_stage = defaultdict(list)
    for k in sorted(s, key=lambda k: (k[1] == "agent_end_to_end", k)):
        if k in c:
            sp = s[k] / c[k]
            by_stage[k[1]].append(sp)
            print(f"| {k[0]} | {k[1]} | {s[k]:.3f} | {c[k]:.3f} | {sp:.2f}× |")
    print("\n| Stage | Geometric-mean speed-up |\n|---|---|")
    for st, v in by_stage.items():
        print(f"| {st} | {statistics.geometric_mean(v):.2f}× |")


if __name__ == "__main__":
    main()
