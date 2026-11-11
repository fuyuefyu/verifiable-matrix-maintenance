#!/usr/bin/env python3
"""Remote-only analysis specification for the ICDE04 measurements.

Independent sampling units are seeds. Repeats and rounds are nested.
Unsuccessful/incomplete runs remain in the manifest and are not imputed.
"""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = [101, 211, 307, 401, 503]
SYNTHETIC_METHODS = {"recompute_check", "incremental_check", "incremental_no_check"}
REAL_METHODS = {"recompute_check", "incremental_check", "cached_factorized_check", "cached_factorized_no_check"}


def median_ci(values, rng):
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Nonfinite or empty summary input")
    central = float(np.median(a))
    boot = np.median(a[rng.integers(0, len(a), size=(2000, len(a)))], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median":central,"ci95_low":float(lo),"ci95_high":float(hi),
            "independent_units":len(a),"seed_values":list(map(float,a))}


def load_run(path):
    rows = []
    metadata = None
    terminal = None
    setups = []
    parse_errors = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                item = json.loads(line)
            except Exception as error:
                parse_errors.append({"line":line_no,"error":str(error)})
                continue
            kind = item.get("kind")
            if terminal is not None:
                parse_errors.append({"line":line_no,"error":"record after terminal","kind":kind})
            if kind == "metadata":
                if metadata is not None:
                    parse_errors.append({"line":line_no,"error":"duplicate metadata"})
                metadata = item
            elif kind == "round":
                rows.append(item)
            elif kind == "setup":
                setups.append(item)
            elif kind in ("completed","failed"):
                if terminal is None:
                    terminal = item
                else:
                    parse_errors.append({"line":line_no,"error":"duplicate terminal","terminal":item})
    return metadata, rows, setups, terminal, parse_errors


def inspect(path, metadata, rows, terminal, parse_errors):
    errors = list(parse_errors)
    if metadata is None:
        errors.append({"error":"missing metadata"})
        return errors
    args = metadata.get("args",{})
    version = metadata.get("version","")
    real = "covertype" in version.lower()
    q61 = version == "q61-field-bench-v1"
    methods = REAL_METHODS if real else SYNTHETIC_METHODS
    if not (real or q61 or version == "exact-field-bench-v2"):
        errors.append({"error":"unsupported run schema","version":version})
    required_flags = ("full_AB_match","oracle_checked","common_A_match","state_match") if q61 else ("full_AB_match","python_bigint_spots_match")
    seeds = metadata.get("selected_seed_ids",[])
    if not seeds:
        errors.append({"error":"missing selected_seed_ids"})
    if terminal is None or terminal.get("kind") != "completed":
        errors.append({"error":"not completed","terminal":terminal})
    identities = set()
    digest_groups = defaultdict(set)
    found_methods = set()
    for row in rows:
        identity = (row.get("seed"),row.get("repeat"),row.get("round"),row.get("method"))
        if identity in identities:
            errors.append({"error":"duplicate round","identity":identity})
        identities.add(identity)
        found_methods.add(row.get("method"))
        if row.get("accepted") is not True or row.get("trial_valid") is not True:
            errors.append({"error":"invalid or rejected round","identity":identity})
        for key in required_flags:
            if row.get(key) is not True:
                errors.append({"error":"correctness flag failed","flag":key,"identity":identity})
        parts = row.get("component_phase_ns",row.get("host_observed_phase_ns"))
        if parts is not None and sum(parts.values()) != row.get("total_ns"):
            errors.append({"error":"time components do not sum","identity":identity})
        if not isinstance(row.get("total_ns"),int) or row["total_ns"] <= 0:
            errors.append({"error":"invalid total time","identity":identity})
        digest = row.get("candidate_sha256")
        if not q61:
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                errors.append({"error":"missing or invalid candidate digest","identity":identity})
            context = row.get("candidate_context")
            context_keys = ("seed","repeat","round","n","t","b" if real else "r")
            if not isinstance(context,dict) or any(k not in context or k not in row or context[k]!=row[k] for k in context_keys):
                errors.append({"error":"missing or inconsistent candidate context","identity":identity})
        if digest:
            digest_groups[identity[:3]].add(digest)
        if row.get("timing_mode") != metadata.get("timing_mode"):
            errors.append({"error":"timing-mode disagreement","identity":identity})
    if found_methods != methods:
        errors.append({"error":"method set mismatch","expected":sorted(methods),"found":sorted(found_methods,key=str)})
    for identity, digests in digest_groups.items():
        if len(digests) != 1:
            errors.append({"error":"candidate differs across matched methods","identity":identity})
    expected = {(s,r,j,m) for s in seeds for r in range(int(args.get("repeats",0)))
                for j in range(int(args.get("rounds",0))) for m in methods}
    if identities != expected:
        errors.append({"error":"round coverage mismatch","expected_count":len(expected),
                       "actual_count":len(identities),"missing_count":len(expected-identities),
                       "unexpected_count":len(identities-expected)})
    if terminal and terminal.get("round_records") != len(rows):
        errors.append({"error":"terminal round count mismatch"})
    if real:
        by_trial=defaultdict(list)
        for row in rows:
            by_trial[(row.get("seed"),row.get("repeat"),row.get("method"))].append(row)
        for identity, trial in by_trial.items():
            running=0
            setup=None
            for row in sorted(trial,key=lambda x:x.get("round",-1)):
                running+=row.get("total_ns",0)
                if setup is None:
                    setup=row.get("production_setup_ns")
                if (not isinstance(setup,int) or setup<=0 or row.get("production_setup_ns")!=setup
                    or row.get("cumulative_update_ns")!=running
                    or row.get("cumulative_production_ns")!=setup+running):
                    errors.append({"error":"cumulative production accounting mismatch","identity":identity,"round":row.get("round")})
    return errors


def summarize(metadata, rows, rng):
    nested = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    phases = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
    memory = defaultdict(list)
    transfers = defaultdict(lambda:defaultdict(list))
    cumulative=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for row in rows:
        method,seed,rep=row["method"],row["seed"],row["repeat"]
        nested[method][seed][rep].append(row["total_ns"]/1e6)
        if "cumulative_production_ns" in row:
            cumulative[method][seed][rep][0]=row["production_setup_ns"]/1e6
            cumulative[method][seed][rep][row["round"]+1]=row["cumulative_production_ns"]/1e6
        for phase, value in row.get("component_phase_ns",{}).items():
            phases[method][phase][seed][rep].append(value/1e6)
        if "gpu_array_peak_bytes" in row.get("memory_after",{}):
            memory[method].append(row["memory_after"]["gpu_array_peak_bytes"])
        for direction in ("h2d_payload_bytes","d2h_payload_bytes"):
            if direction in row:
                transfers[method][direction].append(row[direction])
    seed_ids = sorted(metadata["selected_seed_ids"])
    result = {"metadata":metadata,"seed_order":seed_ids,"methods":{},"paired_speedup":{},"paired_ratios":{}}
    seed_values = {}
    for method in sorted(nested):
        values = [statistics.median(statistics.median(x) for x in nested[method][s].values())
                  for s in seed_ids]
        seed_values[method] = values
        info = median_ci(values,rng)
        info["unit"]="milliseconds per round"
        info["summary_order"]="median rounds within repeat; median repeats within seed; median seeds"
        if memory[method]:
            info["maximum_observed_gpu_live_array_peak_bytes"]=max(memory[method])
            info["memory_note"]="Includes benchmark reset snapshots; reserved pool and RSS HWM are different metrics."
        info["median_payload_bytes"]={key:statistics.median(vals) for key,vals in transfers[method].items()}
        info["synchronized_components_ms"]={}
        for phase, by_seed in phases[method].items():
            vals=[statistics.median(statistics.median(x) for x in by_seed[s].values()) for s in seed_ids]
            info["synchronized_components_ms"][phase]=median_ci(vals,rng)
        result["methods"][method]=info
    if "recompute_check" in seed_values:
        base=seed_values["recompute_check"]
        for method, vals in seed_values.items():
            if method != "recompute_check":
                result["paired_speedup"][method]=median_ci([a/b for a,b in zip(base,vals)],rng)
    ratio_pairs = {
        "incremental_verification_factor": ("incremental_check", "incremental_no_check"),
        "cached_verification_factor": ("cached_factorized_check", "cached_factorized_no_check"),
        "structure_cache_speedup": ("incremental_check", "cached_factorized_check"),
    }
    for name, (numerator, denominator) in ratio_pairs.items():
        if numerator in seed_values and denominator in seed_values:
            stat = median_ci([a/b for a,b in zip(seed_values[numerator],seed_values[denominator])],rng)
            stat.update(numerator=numerator, denominator=denominator,
                        unit="paired time ratio", summary_order="within-seed ratio; median seeds")
            result["paired_ratios"][name] = stat
    result["cumulative_production_ms"]={}
    for method, by_seed in cumulative.items():
        rounds=sorted(next(iter(next(iter(by_seed.values())).values())))
        curves=np.asarray([[statistics.median(rep[j] for rep in by_seed[s].values())
                            for j in rounds] for s in seed_ids],dtype=np.float64)
        indices=rng.integers(0,len(seed_ids),size=(2000,len(seed_ids)))
        boot=np.median(curves[indices],axis=1)
        lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
        result["cumulative_production_ms"][method]={"rounds_completed":rounds,
            "median":np.median(curves,axis=0).tolist(),"ci95_low":lo.tolist(),"ci95_high":hi.tolist(),
            "seed_curves":curves.tolist(),"independent_units":len(seed_ids),
            "interval_scope":"pointwise 95% seed-bootstrap; common resamples across curve points"}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-root",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--require-complete",action="store_true",
                        help="Fail after writing reports if any run is incomplete or no runs exist")
    args=parser.parse_args()
    root=Path(args.raw_root)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261110)
    manifest=[]
    summaries=[]
    candidate_groups=defaultdict(set)
    candidate_sources=defaultdict(set)
    for path in sorted(root.rglob("*.jsonl")):
        metadata,rows,setups,terminal,parse_errors=load_run(path)
        errors=inspect(path,metadata,rows,terminal,parse_errors)
        eligible=bool(not errors and sorted(metadata.get("selected_seed_ids",[]))==SEEDS
                      and metadata.get("args",{}).get("repeats")==3)
        item={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "round_records":len(rows),"errors":errors,"complete":not errors,
              "five_seed_three_repeat_eligible":eligible,
              "primary_latency_eligible":bool(eligible and metadata.get("timing_mode")=="primary_e2e"),
              "metadata":metadata,"terminal":terminal}
        manifest.append(item)
        if not errors:
            result=summarize(metadata,rows,rng)
            result.update(source=str(path),source_sha256=item["sha256"],eligible=eligible,
                          primary_latency_eligible=item["primary_latency_eligible"],setups=setups)
            summaries.append(result)
            for row in rows:
                digest=row.get("candidate_sha256")
                if digest and row.get("candidate_context"):
                    key=(metadata.get("version"),metadata.get("q"),json.dumps(row["candidate_context"],sort_keys=True,separators=(",",":")))
                    candidate_groups[key].add(digest)
                    candidate_sources[key].add(str(path))
    cross_device_mismatches=[{"context":list(key),"digests":sorted(vals),"sources":sorted(candidate_sources[key])}
                             for key,vals in candidate_groups.items() if len(vals)>1]
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    verification={"run_files":len(manifest),"complete_run_files":sum(x["complete"] for x in manifest),
                  "eligible_run_files":sum(x["five_seed_three_repeat_eligible"] for x in manifest),
                  "round_records":sum(x["round_records"] for x in manifest),
                  "candidate_context_groups":len(candidate_groups),
                  "cross_run_candidate_mismatches":cross_device_mismatches,
                  "incomplete_runs_retained":sum(not x["complete"] for x in manifest),
                  "statistics":"2000 bootstrap resamples of the seed-level units reported per run; primary eligible runs have five seeds; paired ratios are formed within seed; repeats/rounds are nested"}
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    with (out/"latency.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["source","version","host","backend","device","q","n","r","batch_size","t","rounds","timing_mode","method","median_ms","ci95_low_ms","ci95_high_ms","seeds","eligible","primary_latency_eligible"])
        writer.writeheader()
        for run in summaries:
            m=run["metadata"]; a=m["args"]
            for method,stat in run["methods"].items():
                writer.writerow(dict(source=run["source"],version=m.get("version"),host=m.get("hostname"),backend=a.get("backend"),device=a.get("device"),q=m.get("q"),n=a.get("n"),r=a.get("r"),batch_size=a.get("batch_size"),t=a.get("t"),rounds=a.get("rounds"),timing_mode=m.get("timing_mode"),method=method,median_ms=stat["median"],ci95_low_ms=stat["ci95_low"],ci95_high_ms=stat["ci95_high"],seeds=stat["independent_units"],eligible=run["eligible"],primary_latency_eligible=run["primary_latency_eligible"]))
    print(json.dumps(verification,sort_keys=True))
    if cross_device_mismatches:
        raise SystemExit("Candidate mismatch across matched run contexts; inspect before publication")
    if args.require_complete and (not manifest or any(not item["complete"] for item in manifest)):
        raise SystemExit("Run collection is empty or contains incomplete/invalid runs; reports retained")


if __name__ == "__main__":
    main()
