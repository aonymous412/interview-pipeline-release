import json, glob, collections, statistics
import pathlib  # [relocated]
import numpy as np

B, SEED = 10000, 42

def boot_ci(vals, B=B, seed=SEED):
    """IC bootstrap percentile -- STRICTEMENT la meme methode que evaluate.py,
    qui a produit les chiffres publies. Une implementation stdlib
    (random.Random + indexation de la liste triee) donne des bornes decalees
    de ~0.1 pt : RNG different et interpolation de percentile differente.
    C'est ce qui avait produit Medical [89.1-91.8] la ou le papier porte
    [89.0-91.7]. Ne pas re-diverger."""
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    boot_means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(B)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return lo, hi

# ---- baselines
base = {}
_ROOT = pathlib.Path(__file__).resolve().parents[2]  # [relocated]
for f in sorted(glob.glob(str(_ROOT / 'results' / 'baselines' / 'results_*_full.json'))):
    d = json.load(open(f))
    m = d[0]['model']
    by = collections.defaultdict(list)
    for r in d:
        by[r['domain']].append(r)
    base[m] = by

# ---- ALI v1
ali = collections.defaultdict(list)
DOMMAP = {'telos': 'Consulting', 'radassist': 'Medical', 'finagent': 'Payments'}
for r in json.load(open(_ROOT / 'results' / 'ali' / 'results_ALI_v1.json')):  # [relocated]
    dom = r.get('domain') or DOMMAP.get(r.get('deployment'), r.get('deployment'))
    ali[dom].append(r)

DOMS = ['Consulting', 'Medical', 'Payments']
print('ALI domain keys found:', {k: len(v) for k, v in ali.items()})
print()

rows = []
for m, by in base.items():
    for dom in DOMS:
        rs = by.get(dom, [])
        if not rs:
            continue
        cov = [r['final_cov'] * 100 for r in rs]
        lo, hi = boot_ci(cov)
        turns = [r['n_turns'] for r in rs]
        stops = collections.Counter(r['stopped_by'] for r in rs)
        capped = sum(v for k, v in stops.items() if k != 'DONE')
        rows.append((m, dom, len(rs), sum(cov)/len(cov), lo, hi,
                     sum(turns)/len(turns), max(turns), capped, dict(stops)))

print(f"{'model':22s} {'domain':11s} {'n':>4s} {'cov%':>6s} {'95% CI':>16s} {'turns':>6s} {'max':>4s} {'non-DONE':>9s}")
for m, dom, n, cov, lo, hi, t, mx, capped, stops in rows:
    print(f'{m:22s} {dom:11s} {n:4d} {cov:6.1f} [{lo:5.1f}-{hi:5.1f}] {t:6.2f} {mx:4d} {capped:9d}')
print()

# ALI per domain
print('--- ALI_v1 per domain')
ali_stats = {}
for dom in DOMS:
    rs = ali.get(dom, [])
    cov = [r['final_cov'] * 100 for r in rs]
    lo, hi = boot_ci(cov)
    turns = [r['n_turns'] for r in rs]
    ali_stats[dom] = (len(rs), sum(cov)/len(cov), lo, hi, sum(turns)/len(turns))
    print(f'{dom:11s} n={len(rs):3d} cov={sum(cov)/len(cov):5.1f} [{lo:5.1f}-{hi:5.1f}] turns={sum(turns)/len(turns):5.2f}')
print()

# gap vs best baseline per domain (all models, and n=150 models only)
print('--- gap ALI_v1 vs best baseline, per domain')
for dom in DOMS:
    cands = [(cov, m, n) for (m, d, n, cov, *_ ) in rows if d == dom]
    best = max(cands)
    big = max([(cov, m, n) for (cov, m, n) in cands if n >= 50] or cands)
    a = ali_stats[dom][1]
    print(f'{dom:11s} ALI={a:5.1f}  best-any={best[0]:5.1f} ({best[1]}, n={best[2]}) gap=+{a-best[0]:4.1f}   '
          f'best-n50={big[0]:5.1f} ({big[1]}) gap=+{a-big[0]:4.1f}')
print()

# leave-Payments-out sensitivity, Opus vs ALI (both n=150, 50/domain)
print('--- sensitivity: drop Payments (the low-kappa domain)')
opus = base['claude-opus-4-8']
for label, doms in [('all 3 domains', DOMS), ('Consulting+Medical only', ['Consulting', 'Medical'])]:
    o = [r['final_cov']*100 for d in doms for r in opus[d]]
    a = [r['final_cov']*100 for d in doms for r in ali[d]]
    olo, ohi = boot_ci(o); alo, ahi = boot_ci(a)
    print(f'{label:24s} Opus={sum(o)/len(o):5.2f} [{olo:.2f}-{ohi:.2f}] (n={len(o)})  '
          f'ALI={sum(a)/len(a):5.2f} [{alo:.2f}-{ahi:.2f}] (n={len(a)})  gap=+{sum(a)/len(a)-sum(o)/len(o):5.2f}pp')
print()

# global stop-reason tally
print('--- stop reasons, all baseline runs pooled')
allstops = collections.Counter()
for m, by in base.items():
    for dom in DOMS:
        for r in by.get(dom, []):
            allstops[r['stopped_by']] += 1
tot = sum(allstops.values())
print(f'total baseline runs: {tot}  ->  {dict(allstops)}')
print(f"self-declared DONE: {allstops['DONE']}/{tot} = {allstops['DONE']/tot*100:.1f}%")
