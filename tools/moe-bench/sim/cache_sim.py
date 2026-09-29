#!/usr/bin/env python3
"""cache_sim.py trace.txt [...] --frac 0.25,0.35,0.5: hit rates of per-layer expert caches on a GGML_MOE_LOG routing trace.

Trace lines: "<tensor name> id0 id1 ...", one per ffn_gate_exps op (one token, one layer); concatenated in the order given
(topic switches included). Each layer has its own cache of C = frac * n_experts slots (the fork's layout). Policies:
  lru     least recently used
  lfu     least frequently used (lifetime counts, no decay)
  lfuw    LFU with window decay (counts halve every 512 tokens)
  hot     static: the C most used experts of the WHOLE trace (oracle profile, like a perfect persistent-hot profile)
  belady  offline optimal: evict the expert whose next use is farthest (upper bound for any policy, incl. any predictor)
A miss inserts the expert (evicting per policy); a token's experts are all looked up first, then inserted."""
import sys, re, argparse, collections, heapq

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--frac", default="0.25,0.35,0.5")
ap.add_argument("--skip", type=int, default=0, help="ignore the first N tokens of each file (prompt processing)")
ap.add_argument("--sweep", action="store_true", help="sweep the fork policy window / global weight / margin / inserts at the first --frac")
ap.add_argument("--warm", type=int, default=0, help="simulate from token 0 but count hits only from token N of the concatenated trace (steady state)")
a = ap.parse_args()
per = collections.defaultdict(list)  # layer -> list of tuples (token order)
n_exp = 0
for fn in a.files:
    lines = open(fn).read().splitlines()
    cnt = collections.Counter()
    for l in lines:
        p = l.split()
        m = re.search(r"(\d+)", p[0]); il = int(m.group(1))
        ids = [int(x) for x in p[1:] if int(x) >= 0]
        if cnt[il] >= a.skip: per[il].append(tuple(ids))
        cnt[il] += 1
        n_exp = max(n_exp, max(ids) + 1 if ids else 0)
n_exp = 1 << (n_exp - 1).bit_length()
layers = sorted(per)
print(f"{len(layers)} layers, {len(per[layers[0]])} tokens, {n_exp} experts (rounded), top-{len(per[layers[0]][0])}")

def sim_lru(seq, C, max_ins=None):
    cache = collections.OrderedDict(); hit = tot = 0; t_ = 0
    for ids in seq:
        for e in ids:
            if t_ >= a.warm:
                tot += 1
                if e in cache: hit += 1
        t_ += 1
        ins_ = 0
        for e in ids:
            if e in cache: cache.move_to_end(e)
            elif max_ins is not None and ins_ >= max_ins: continue
            else:
                ins_ += 1
                if len(cache) >= C:
                    for v in cache:
                        if v not in ids: del cache[v]; break
                    else: continue
                cache[e] = 1
    return hit, tot

def sim_lfu(seq, C, decay=0):
    cnt = collections.Counter(); cache = set(); hit = tot = 0
    for t, ids in enumerate(seq):
        if decay and t % decay == 0 and t:
            for k in list(cnt): cnt[k] //= 2
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        for e in ids: cnt[e] += 1
        for e in ids:
            if e in cache: continue
            if len(cache) >= C:
                v = min((x for x in cache if x not in ids), key=lambda x: cnt[x], default=None)
                if v is None or cnt[v] > cnt[e]: continue
                cache.remove(v)
            cache.add(e)
    return hit, tot


def sim_fork(seq, C, window=64, k=16.0, margin=2.0, max_ins=2):
    """the fork's default policy (LLAMA_MOE_CACHE_POLICY=add): score = uses in the last `window` tokens + k * lifetime/lifetime_max;
    a miss replaces the lowest-scored cached expert not used this token if its score exceeds the victim's + margin; at most
    max_ins inserts per layer and token; new experts are usable from the next token."""
    glob = collections.Counter(); win = collections.Counter(); recent = collections.deque(); cache = set(); hit = tot = 0; gmax = 1
    for t, ids in enumerate(seq):
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        if len(recent) >= window:
            for e in recent.popleft(): win[e] -= 1
        recent.append(ids)
        for e in ids:
            win[e] += 1; glob[e] += 1; gmax = max(gmax, glob[e])
        sc = lambda x: win[x] + k*glob[x]/gmax
        ins = 0
        for e in sorted((x for x in ids if x not in cache), key=lambda x: -sc(x)):
            if ins >= max_ins: break
            if len(cache) < C: cache.add(e); ins += 1; continue
            v = min((x for x in cache if x not in ids), key=sc, default=None)
            if v is None or sc(e) <= sc(v) + margin: continue
            cache.remove(v); cache.add(e); ins += 1
    return hit, tot

def sim_hot(seq, C):
    cnt = collections.Counter(e for ids in seq for e in ids)
    cache = {e for e, _ in cnt.most_common(C)}
    hit = sum(1 for ids in seq[a.warm:] for e in ids if e in cache); return hit, sum(len(i) for i in seq[a.warm:])

def sim_belady(seq, C):
    nxt = collections.defaultdict(collections.deque)
    for t, ids in enumerate(seq):
        for e in ids: nxt[e].append(t)
    cache = {}; hit = tot = 0  # expert -> next use time
    INF = 1 << 60
    for t, ids in enumerate(seq):
        for e in ids:
            nxt[e].popleft()
        for e in ids:
            if t >= a.warm:
                tot += 1
                if e in cache: hit += 1
        for e in ids:
            n = nxt[e][0] if nxt[e] else INF
            if e in cache: cache[e] = n; continue
            if len(cache) >= C:
                v = max((x for x in cache if x not in ids), key=lambda x: cache[x], default=None)
                if v is None or cache[v] <= n: continue  # bypass: the newcomer is needed later than everything cached
                del cache[v]
            cache[e] = n
    return hit, tot

for f in [float(x) for x in a.frac.split(",")]:
    C = max(1, int(f*n_exp)); res = {}
    for name, fn in [("lru", sim_lru), ("lru2", lambda s, c: sim_lru(s, c, 2)), ("lfu", lambda s, c: sim_lfu(s, c)), ("lfuw", lambda s, c: sim_lfu(s, c, 512)), ("fork", sim_fork), ("hot", sim_hot), ("belady", sim_belady)]:
        h = t = 0
        for il in layers:
            x, y = fn(per[il], C); h += x; t += y
        res[name] = 100.0*h/t
    print(f"cache {int(100*f):3d}% ({C} slots/layer): " + "  ".join(f"{k} {v:5.1f}%" for k, v in res.items()) +
          f"   | misses vs fork: lru {100*(1-res['lru']/100)/(1-res['fork']/100):.0f}%, lru2 {100*(1-res['lru2']/100)/(1-res['fork']/100):.0f}%, belady {100*(1-res['belady']/100)/(1-res['fork']/100):.0f}%")

if a.sweep:
    f = float(a.frac.split(",")[0]); C = max(1, int(f*n_exp)); rows = []
    for w in (4, 8, 16, 32, 64, 128):
        for k in (0.0, 2.0, 8.0, 16.0):
            for mg in (0.0, 2.0):
                for mi in (2, 4):
                    h = t = 0
                    for il in layers:
                        x, y = sim_fork(per[il], C, window=w, k=k, margin=mg, max_ins=mi); h += x; t += y
                    rows.append((100.0*h/t, w, k, mg, mi))
    rows.sort(reverse=True)
    print(f"sweep at {int(100*f)}% cache ({C} slots): best 8 and the default (window 64, k 16, margin 2, inserts 2)")
    for r in rows[:8]: print(f"  {r[0]:5.1f}%  window {r[1]:3d}  k {r[2]:4.1f}  margin {r[3]:.0f}  inserts {r[4]}")
    d = [r for r in rows if r[1:] == (64, 16.0, 2.0, 2)][0]; print(f"  default {d[0]:5.1f}%  (rank {rows.index(d) + 1} of {len(rows)})")
