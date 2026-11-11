#!/usr/bin/env python3
"""Web-authored specification; run only on an authorized remote worker.

Computes analytical upper bounds, never empirical attack probabilities.
Uses exact fractions for validation and small illustrative profiles.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import socket
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


def rank_counts(n, r, q):
    out = []
    for s in range(min(n, r) + 1):
        num = den = 1
        for i in range(s):
            num *= (q**n - q**i) * (q**r - q**i)
            den *= q**s - q**i
        value, rem = divmod(num, den)
        assert rem == 0
        out.append(value)
    assert sum(out) == q**(n*r)
    return out


def mu(n, r, q, w):
    counts = rank_counts(n, r, q)
    return sum((Fraction(c, q**(n*r + max(n*s-w, 0)))
                for s, c in enumerate(counts)), Fraction(0))


def agreement(n, r, q, w, R0=None):
    R0 = n*n if R0 is None else R0
    if w >= R0:
        return Fraction(1)
    theta = Fraction(2*q-1, q*q)
    return min(theta, mu(n, r, q, w))


def accept(n, r, q, w, t, R0=None):
    p = agreement(n, r, q, w, R0)
    return p + (1-p)*Fraction(1, q**t)


def actions(n, r, R0, budget):
    stop = min(n*min(n, r), R0-1, budget)
    out = list(range(stop+1))
    if R0 <= budget:
        out.append(R0)
    return out


def dp(n, r, q, times, W, compressed=True):
    R0 = n*n
    prev = [Fraction(1)]*(W+1)
    arglayers = []
    for t in reversed(times):
        all_a = {w: accept(n, r, q, w, t, R0)
                 for w in (actions(n, r, R0, W) if compressed else range(W+1))}
        curr, args = [], []
        for z in range(W+1):
            candidates = actions(n, r, R0, z) if compressed else range(z+1)
            best, arg = None, None
            for w in candidates:
                score = all_a[w]*prev[z-w]
                if best is None or score > best:
                    best, arg = score, w
            curr.append(best)
            args.append(arg)
        prev = curr
        arglayers.append(args)
    allocations = []
    for budget in range(W+1):
        z, alloc = budget, []
        for args in reversed(arglayers):
            w = args[z]
            alloc.append(w)
            z -= w
        allocations.append(alloc)
    return prev, allocations


def compositions_at_most(k, W, prefix=()):
    if k == 0:
        yield prefix
        return
    for w in range(W+1):
        yield from compositions_at_most(k-1, W-w, prefix+(w,))


def matrix_rank(flat, n, r, q):
    a = [list(flat[i*r:(i+1)*r]) for i in range(n)]
    row = 0
    for col in range(r):
        pivot = next((j for j in range(row, n) if a[j][col] % q), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        inv = pow(a[row][col], -1, q)
        a[row] = [(x*inv) % q for x in a[row]]
        for j in range(n):
            if j != row:
                c = a[j][col]
                a[j] = [(a[j][h]-c*a[row][h]) % q for h in range(r)]
        row += 1
        if row == n:
            break
    return row


def fmt(x):
    with localcontext() as ctx:
        ctx.prec = 100
        d = Decimal(x.numerator)/Decimal(x.denominator)
        return str(d), str(d.log10()) if d else "-Infinity"


def validate(deadline):
    rank_matrices = 0
    rank_cases = []
    for n, r, q in ((2,2,2), (2,2,3), (3,2,2), (4,2,2)):
        observed = [0]*(r+1)
        for flat in itertools.product(range(q), repeat=n*r):
            observed[matrix_rank(flat, n, r, q)] += 1
            rank_matrices += 1
        assert observed == rank_counts(n, r, q)
        rank_cases.append({"n":n,"r":r,"q":q,"counts":observed})
        deadline()
    allocation_comparisons = 0
    configurations = 0
    balanced_comparisons = 0
    for n, r, q in ((2,2,2), (3,2,2), (3,2,3), (4,2,2)):
        for times in ((1,1), (1,2), (2,1,2)):
            Wmax = min(n*n+2, 12)
            compressed, args = dp(n,r,q,times,Wmax,True)
            full, _ = dp(n,r,q,times,Wmax,False)
            assert compressed == full
            for W in range(Wmax+1):
                brute = Fraction(0)
                for allocation in compositions_at_most(len(times), W):
                    value = Fraction(1)
                    for w,t in zip(allocation,times):
                        value *= accept(n,r,q,w,t)
                    brute = max(brute,value)
                    allocation_comparisons += 1
                assert brute == compressed[W]
                assert sum(args[W]) <= W
                configurations += 1
                deadline()
        for k in (2,3):
            for extra in range(4):
                T = k+extra
                ell, s = divmod(T,k)
                balanced = [ell+1]*s+[ell]*(k-s)
                for W in (0, n*r, n*n):
                    target = dp(n,r,q,balanced,W)[0][W]
                    for increments in compositions_at_most(k, extra):
                        if sum(increments) != extra:
                            continue
                        times = [1+x for x in increments]
                        value = dp(n,r,q,times,W)[0][W]
                        assert target <= value
                        balanced_comparisons += 1
                    deadline()
    assert mu(4,2,2,0) == Fraction(593,32768)
    assert mu(4,2,2,7) == Fraction(151,256)
    assert mu(4,2,2,8) == 1
    vals, args = dp(4,2,2,[1]*4,8)
    expected = Fraction(7,8)*Fraction(33361,65536)**3
    assert vals[8] == expected
    assert sorted(args[8]) == [0,0,0,8]
    return {"rank_matrices":rank_matrices,"rank_cases":rank_cases,
            "allocation_comparisons":allocation_comparisons,
            "configurations":configurations,
            "balanced_comparisons":balanced_comparisons,
            "mismatches":0,
            "example":{"n":4,"r":2,"q":2,"k":4,"t":1,"W":8,
                       "profile_fraction":str(expected),
                       "threshold_fraction":str(Fraction(7,8)**4),
                       "profile_decimal":fmt(expected)[0],
                       "maximizing_envelope_allocation":args[8]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    parser.add_argument("--max-seconds",type=int,default=1200)
    args = parser.parse_args()
    start = time.time()
    hard_stop = 1794317534  # 2026-11-10T13:32:14Z; checked by executor too.
    stop = min(start+args.max_seconds, hard_stop)
    def deadline():
        if time.time() >= stop:
            raise TimeoutError("analytical bound computation deadline")
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    result = validate(deadline)
    with (out/"work_profile.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["n","r","q","w","R0","w_over_nr","mu","mu_log10","pbar","pbar_log10","theta","kind"])
        writer.writeheader()
        for n,r,q in ((4,2,2),(8,2,2),(16,4,17),(64,8,65521)):
            for w in sorted(set(range(0,n*r+1,max(1,n*r//64)))|{n*r,n*n-1,n*n}):
                m,p=mu(n,r,q,w),agreement(n,r,q,w)
                mv,ml=fmt(m); pv,pl=fmt(p)
                writer.writerow(dict(n=n,r=r,q=q,w=w,R0=n*n,w_over_nr=w/(n*r),mu=mv,mu_log10=ml,pbar=pv,pbar_log10=pl,theta=fmt(Fraction(2*q-1,q*q))[0],kind="analytical_upper_bound"))
            deadline()
    with (out/"aggregate_profile.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["n","r","q","k","t","W","R0","profile_bound","profile_log10","threshold_bound","threshold_log10","allocation","kind"])
        writer.writeheader()
        for n,r,q,k,t in ((4,2,2,4,1),(4,2,2,8,2),(8,2,2,4,1),(8,2,17,4,2),(16,4,17,4,2)):
            Wmax=min(k*n*n,256)
            vals,allocs=dp(n,r,q,[t]*k,Wmax)
            a=Fraction(2*q-1,q*q)+(1-Fraction(2*q-1,q*q))*Fraction(1,q**t)
            for W in range(Wmax+1):
                old=a**(k-min(k,W//(n*n)))
                assert vals[W] <= old
                pv,pl=fmt(vals[W]); ov,ol=fmt(old)
                writer.writerow(dict(n=n,r=r,q=q,k=k,t=t,W=W,R0=n*n,profile_bound=pv,profile_log10=pl,threshold_bound=ov,threshold_log10=ol,allocation=json.dumps(allocs[W],separators=(",",":")),kind="analytical_upper_bound"))
            deadline()
    result.update(hostname=socket.gethostname(),platform=platform.platform(),python=platform.python_version(),pid=os.getpid(),start_unix=start,finish_unix=time.time(),elapsed_seconds=time.time()-start,interpretation="Exact bounds and optimizer checks; not attack measurements.")
    (out/"profile_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/"sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
