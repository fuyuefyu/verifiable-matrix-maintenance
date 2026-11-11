#!/usr/bin/env python3
"""Pro-authored exact analytical planning; execute only on an authorized remote.

The output consists of upper certificates, never measured attack probabilities.
The exact planner uses integer breakpoints; full enumeration validates the reduction.
All scientific arithmetic is performed by this file on the remote worker.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import socket
import sys
import time
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

sys.set_int_max_str_digits(100000)
HARD_STOP = 1794317534


def decimal(x, precision=70):
    with localcontext() as ctx:
        ctx.prec = precision
        return str(Decimal(x.numerator) / Decimal(x.denominator))


def log10(x):
    with localcontext() as ctx:
        ctx.prec = 70
        return str((Decimal(x.numerator) / Decimal(x.denominator)).log10())


def exact_record(x):
    return {"numerator": str(x.numerator), "denominator": str(x.denominator),
            "decimal": decimal(x), "log10": log10(x) if x else None}


def alpha(theta, q, t):
    return theta + (1-theta) / q**t


def certificate(theta, q, checks, b):
    vals = sorted((alpha(h, q, t) for h, t in zip(theta, checks)), reverse=True)
    return math.prod(vals[:len(vals)-b], start=F(1))


class TwoGroup:
    """Exact reduction by convexity and symmetry within each theta group.

    There are m rounds in each group. The top 2m-b alpha values contain all
    high-theta rounds and m-b low-theta rounds because theta_A > alpha_D(1).
    An optimum exists with within-group checking counts differing by at most one.
    """
    def __init__(self, q, h=F(15,16), d=F(8,9), m=256, b=128, t_min=1):
        self.q, self.h, self.d, self.m, self.b, self.t_min = q,h,d,m,b,t_min
        assert m>=1 and t_min>=1 and 0 <= b <= m and h > alpha(d,q,t_min)
        self.cache = {}
        self.coefficients = {}

    def a(self, theta, t):
        key = theta,t
        if key not in self.coefficients:
            self.coefficients[key] = alpha(theta,self.q,t)
        return self.coefficients[key]

    def score(self, extra_a, extra_d):
        m = self.m
        la, sa = divmod(extra_a, m)
        ld, sd = divmod(extra_d, m)
        la += self.t_min
        ld += self.t_min
        retained = m-self.b
        d_low_checks = min(retained,m-sd)
        return (self.a(self.h,la)**(m-sa) * self.a(self.h,la+1)**sa *
                self.a(self.d,ld)**d_low_checks *
                self.a(self.d,ld+1)**(retained-d_low_checks))

    def schedule(self, extra_a, extra_d):
        la,sa=divmod(extra_a,self.m)
        ld,sd=divmod(extra_d,self.m)
        aa=[self.t_min+la+(i<sa) for i in range(self.m)]
        dd=[self.t_min+ld+(i<sd) for i in range(self.m)]
        return [t for pair in zip(aa,dd) for t in pair]

    def breakpoint_candidates(self, extra):
        # log(score(sa, extra-sa)) is affine between these integer knots.
        points={0,extra}
        for h in range(extra//self.m+2):
            for candidate in (h*self.m,extra-h*self.m,extra-(h*self.m+self.b)):
                if 0<=candidate<=extra:
                    points.add(candidate)
        return sorted(points)

    def optimum_full(self, total):
        extra=total-2*self.m*self.t_min
        assert extra>=0
        best,chosen=None,None
        for sa in range(extra+1):
            value=self.score(sa,extra-sa)
            if best is None or value<best:
                best,chosen=value,sa
        return best,chosen,extra-chosen

    def optimum(self, total):
        if total not in self.cache:
            extra=total-2*self.m*self.t_min
            assert extra>=0
            best,chosen=None,None
            for sa in self.breakpoint_candidates(extra):
                value=self.score(sa,extra-sa)
                if best is None or value<best:
                    best,chosen=value,sa
            self.cache[total]=(best,chosen,extra-chosen)
        return self.cache[total]

    def rr(self,total):
        extra=total-2*self.m*self.t_min
        assert extra>=0
        full,rem=divmod(extra,2*self.m)
        sa=full*self.m+(rem+1)//2
        sd=full*self.m+rem//2
        return self.score(sa,sd),sa,sd

    def minimum(self, target, method):
        floor=self.h**self.m*self.d**(self.m-self.b)
        if target<=floor:
            return None
        fn=self.optimum if method=="optimal" else self.rr
        low=2*self.m*self.t_min
        high=low
        if fn(low)[0] > target:
            high=low+1
            while fn(high)[0]>target:
                high=low+2*(high-low)
                assert high<=low+8192
                deadline()
        left=low
        while left<high:
            mid=(left+high)//2
            if fn(mid)[0]<=target:
                high=mid
            else:
                left=mid+1
            deadline()
        value,sa,sd=fn(left)
        assert value<=target
        previous=fn(left-1)[0] if left>low else None
        assert previous is None or previous>target
        schedule=self.schedule(sa,sd)
        # The fixed prefix RR is balanced within each interleaved group too.
        assert sum(schedule)==left
        assert certificate([self.h,self.d]*self.m,self.q,schedule,self.b)==value
        return dict(total_checks=left,extra_a=sa,extra_d=sd,t_schedule=schedule,
                    bound=exact_record(value),
                    previous_budget_bound=exact_record(previous) if previous else None)


def deadline():
    if time.time()>=STOP:
        raise TimeoutError("analytical planning hard deadline")


def counts(n,r,q):
    result=[]
    for s in range(r+1):
        num=den=1
        for i in range(s):
            num*=(q**n-q**i)*(q**r-q**i)
            den*=q**s-q**i
        count,rem=divmod(num,den)
        assert rem==0
        result.append(count)
    assert sum(result)==q**(n*r)
    return result


def uniform_profile(n,r,q,w):
    return sum((F(c,q**(n*r+max(n*s-w,0)))
                for s,c in enumerate(counts(n,r,q))),F(0))


def compositions(total,k):
    if k==1:
        yield (total,)
    else:
        for x in range(total+1):
            for tail in compositions(total-x,k-1):
                yield (x,)+tail


def validate_group_reduction():
    cases=compared=0
    for q in (5,17,257):
        for m in (1,2,3):
            for b in range(m+1):
                model=TwoGroup(q,m=m,b=b)
                for extra in range(6):
                    best=model.optimum(2*m+extra)[0]
                    assert model.optimum(2*m+extra)==model.optimum_full(2*m+extra)
                    brute=None
                    for allocation in compositions(extra,2*m):
                        val=certificate([model.h,model.d]*m,q,
                                        [1+x for x in allocation],b)
                        compared+=1
                        brute=val if brute is None else min(brute,val)
                    assert brute==best
                    cases+=1
                    deadline()
    large=[]
    for q in (5,17,257,65521):
        model=TwoGroup(q)
        for extra in (0,1,127,128,129,254,255,256,257,383,384,511,512,767):
            fast=model.optimum(512+extra)
            full=model.optimum_full(512+extra)
            assert fast==full
            large.append({"q":q,"extra":extra,"full_candidates":extra+1,
                          "breakpoint_candidates":len(model.breakpoint_candidates(extra)),
                          "best_extra_a":fast[1]})
            deadline()
    return dict(cases=cases,allocations_compared=compared,mismatches=0,
                full_size_breakpoint_comparisons=large)


def main():
    global STOP
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--max-seconds",type=int,default=900)
    args=ap.parse_args()
    start=time.time()
    STOP=min(start+args.max_seconds,HARD_STOP)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    verified=validate_group_reduction()
    k,m,b=512,256,128
    target=F(31,50)**64
    model=TwoGroup(17,m=m,b=b)
    planning={}
    for method in ("optimal","round_robin"):
        # Clear caches so one policy is not charged for another's setup.
        run=TwoGroup(17,m=m,b=b)
        began=time.perf_counter_ns()
        plan=run.minimum(target,method)
        elapsed=time.perf_counter_ns()-began
        plan["planning_ns"]=elapsed
        planning[method]=plan
        deadline()
    certs={"target":exact_record(target),"work_budget_gates":128*40*40,
           "R0":40*40,"k":k,"b":b,"exact_planning":planning,
           "meaning":"Upper bound on all rounds accepting under the stated pathwise bilinear gate budget, not a lower bound on attack success.",
           "interpretation":"q17 is a resource-allocation mechanism study; q65521 t2 is the output-integrity service setting."}
    plans=[]
    for method,name in (("optimal","sue_opt17"),("round_robin","sue_rr17")):
        p=planning[method]
        checks=p["t_schedule"]
        plans.append(dict(schema="connected-plan-v1",name=name,purpose="mechanism",
                    encoding="sue",q=17,n=40,r=2,k=k,
                    eta_schedule=["1/4","1/3"]*m,t_schedule=checks,
                    planning_ns=p["planning_ns"],
                    typed_program={"op":"outer_product_sum","n":40,"r":2,
                                   "bilinear_gates_per_round":40*40*2},
                    certificate={"bound_decimal":p["bound"]["decimal"],
                        "target_decimal":decimal(target),"b":b,"R0":1600,
                        "work_budget_gates":204800,"total_checks":sum(checks)},
                    incorrect_increment_union_bound_decimal=decimal(min(F(1),sum((F(1,17**t) for t in checks),F(0))))))
    for encoding,etas in (("sue",[F(1,4),F(1,3)]),("oue",[F(1,10),F(1,5)])):
        theta=[1-eta*eta for eta in etas]*m
        q=65521;checks=[2]*k
        bound=certificate(theta,q,checks,b)
        floor=math.prod(sorted(theta,reverse=True)[:k-b],start=F(1))
        certs[encoding]={"q":q,"t":2,"bound":exact_record(bound),
                         "floor":exact_record(floor),"meets_work_target":bound<=target,
                         "incorrect_increment_union_bound":exact_record(F(k,q*q))}
        plans.append(dict(schema="connected-plan-v1",name=encoding+"_integrity65521",
                    purpose="integrity",encoding=encoding,q=q,n=40,r=2,k=k,
                    eta_schedule=[str(x) for x in etas]*m,t_schedule=checks,
                    planning_ns=0,
                    typed_program={"op":"outer_product_sum","n":40,"r":2,
                                   "bilinear_gates_per_round":40*40*2},
                    certificate={"bound_decimal":decimal(bound),"floor_decimal":decimal(floor),
                        "target_decimal":decimal(target),"b":b,"R0":1600,
                        "work_budget_gates":204800,"total_checks":sum(checks),
                        "meets_work_target":bound<=target},
                    incorrect_increment_union_bound_decimal=decimal(F(k,q*q))))
    assert certs["sue"]["meets_work_target"] and not certs["oue"]["meets_work_target"]
    assert F(15,16)**4*F(8,9)**2==F(625,1024)
    with (out/"allocation_curve.csv").open("w",newline="") as f:
        fields=["q","k","b","total_checks","policy","certificate","log10","target","kind"]
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        selected=sorted(set(range(512,1025,16))|{planning[x]["total_checks"]+d for x in planning for d in (-1,0)})
        for total in selected:
            for name,fn in (("optimal",model.optimum),("fixed_order_round_robin",model.rr)):
                value=fn(total)[0]
                wr.writerow(dict(q=17,k=k,b=b,total_checks=total,policy=name,
                           certificate=decimal(value),log10=log10(value),target=decimal(target),kind="analytical_upper_bound"))
            deadline()
    with (out/"uniform_comparison.csv").open("w",newline="") as f:
        fields=["n","r","q","w","mu","range_bound","cm_geometric_bound","cm_full_bound","theta","kind"]
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        n,r,q=16,4,17;g=n*r-(r*r)//4;mu0=uniform_profile(n,r,q,0)
        for w in range(n*r+1):
            mu=uniform_profile(n,r,q,w)
            rb=min(F(1),q**w*mu0)
            cmg=F(2,q)**max(g-w,0)
            exponent=max(F(n*r-w,3),F(g-w),F(0))
            with localcontext() as ctx:
                ctx.prec=80
                cmfull=(Decimal(2)/q)**(Decimal(exponent.numerator)/exponent.denominator)
            assert mu<=rb
            if 0<w<n*r:
                assert mu<rb
            wr.writerow(dict(n=n,r=r,q=q,w=w,mu=decimal(mu),range_bound=decimal(rb),
                       cm_geometric_bound=decimal(cmg),cm_full_bound=str(cmfull),
                       theta=decimal(F(2*q-1,q*q)),kind="analytical_agreement_upper_bound"))
        deadline()
    with (out/"target_sensitivity.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["q","epoch_target","global_target","policy","checks","bound"]);wr.writeheader()
        for q in (5,17,257,65521):
            mod=TwoGroup(q,m=m,b=b)
            for goal in (F(153,250),F(31,50),F(5,8),F(13,20),F(7,10)):
                for method in ("optimal","round_robin"):
                    p=mod.minimum(goal**64,method)
                    wr.writerow(dict(q=q,epoch_target=str(goal),global_target=decimal(goal**64),
                                    policy=method,checks=None if p is None else p["total_checks"],
                                    bound=None if p is None else p["bound"]["decimal"]))
                deadline()
    (out/"plans.json").write_text(json.dumps({"schema":"connected-plans-v1","plans":plans},indent=2)+"\n")
    (out/"exact_certificates.json").write_text(json.dumps(certs,indent=2)+"\n")
    meta=dict(start_unix=start,finish_unix=time.time(),elapsed_seconds=time.time()-start,
              hostname=socket.gethostname(),platform=platform.platform(),python=platform.python_version(),
              pid=os.getpid(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              planning_algorithm="exact piecewise-log-affine breakpoint enumeration after within-type balancing",
              group_reduction_validation=verified,optimal_total=planning["optimal"]["total_checks"],
              rr_total=planning["round_robin"]["total_checks"],
              planning_ns={x:planning[x]["planning_ns"] for x in planning},
              artifacts={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir()})
    (out/"PLANNER_RECEIPT.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({k:meta[k] for k in ("elapsed_seconds","hostname","optimal_total","rr_total","planning_ns","group_reduction_validation")},indent=2))


if __name__=="__main__":
    main()
