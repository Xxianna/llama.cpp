#!/usr/bin/env python3
"""report.py RES: aggregate results.tsv (machine, test, variant, metric, value, note) into markdown tables (mean, n) per test/metric/variant."""
import sys, os, collections, statistics as st, re
res = sys.argv[1]; rows = []
for f in ("results.tsv", "results_w10.tsv", "results_pc2.tsv"):
    p = os.path.join(res, f)
    if os.path.exists(p):
        for l in open(p):
            x = l.rstrip("\n").split("\t")
            if len(x) >= 6 and x[0] == "RESULT": rows.append(x[1:6])
d = collections.defaultdict(list)
for m, t, v, k, val in rows:
    try: d[(m, t, k, re.sub(r"#\d+$", "", v))].append(float(val))
    except ValueError: pass
print("# Campaign report\n")
cur = None
for key in sorted(d):
    m, t, k, v = key
    if (m, t, k) != cur:
        cur = (m, t, k); print(f"\n## {m} / {t} / {k}\n\n| variant | mean | sd | n |\n|---|---|---|---|")
    vals = d[key]; print(f"| {v} | {st.mean(vals):.3f} | {st.pstdev(vals) if len(vals)>1 else 0:.3f} | {len(vals)} |")
