#!/usr/bin/env python3
"""Pro-authored analysis of the new connected application results only.

Authoring validation is AST-only. Run this script on an authorized analysis host.
It never executes input scripts, changes raw inputs, drops failed units, or pools
live-integrity observations with the small-field replay contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
from decimal import Decimal, localcontext
from fractions import Fraction as F

os.environ.setdefault("MPLBACKEND", "Agg")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "4"
import numpy as np

sys.set_int_max_str_digits(100000)
N, R, K, SCALE = 40, 2, 512, 144
PREFIXES = (16384, 65536, 262144, 581012)
IDS = tuple(range(20))
ENCODINGS = ("sue", "oue")
PLAN_NAMES = ("sue_opt17", "sue_rr17", "sue_integrity65521", "oue_integrity65521")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name):
    value = float(value)
    require(math.isfinite(value), name + " is not finite")
    return value


def close(actual, expected, name, rtol=2e-10, atol=1e-12):
    actual, expected = finite(actual, name), finite(expected, name)
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            "%s mismatch: recorded %r, recomputed %r" % (name, actual, expected))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"shape": list(value.shape), "dtype": value.dtype.str},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    h = hashlib.sha256(header)
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    def forbidden(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=forbidden)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def child(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Expected a relative result filename")
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Result filename escapes its supplied input directory")
    return path


class Inventory:
    def __init__(self):
        self.items = {}

    def add(self, path, role, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), "Missing required input: " + str(path))
        key = str(path)
        if key not in self.items:
            self.items[key] = {"path": key, "bytes": path.stat().st_size,
                               "sha256": sha_file(path), "roles": []}
        item = self.items[key]
        if role not in item["roles"]:
            item["roles"].append(role)
        if expected is not None:
            require(item["sha256"] == expected, "Input SHA256 mismatch: " + key)
        return item["sha256"]

    def records(self):
        return [self.items[key] for key in sorted(self.items)]


def completed_directory(root, mode, inventory, harness_sha=None):
    root = Path(root).resolve()
    inventory.add(root / "status.json", mode + " completion status")
    status = read_json(root / "status.json")
    require(status.get("status") == "completed", "Incomplete input directory: " + str(root))
    require(status.get("args", {}).get("mode") == mode, "Unexpected result mode in " + str(root))
    if harness_sha:
        require(status.get("metadata", {}).get("script_sha256") == harness_sha,
                "Unexpected harness source hash in " + str(root))
    hashes = status.get("files_sha256")
    require(isinstance(hashes, dict) and hashes, "Missing output hash manifest")
    for name, expected in hashes.items():
        inventory.add(child(root, name), mode + " raw output", expected)
    return status


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError("Malformed JSONL %s:%d" % (path, number)) from exc
                yield row


def exact(record):
    return F(int(record["numerator"]), int(record["denominator"]))


def exact_summary(value):
    with localcontext() as ctx:
        ctx.prec = 70
        d = Decimal(value.numerator) / Decimal(value.denominator)
        return {"numerator": str(value.numerator), "denominator": str(value.denominator),
                "decimal": str(d), "log10": str(d.log10()) if value else None}


def expand(values):
    require(isinstance(values, list) and len(values) in (8, K), "Schedule length is not 8 or 512")
    return values * 64 if len(values) == 8 else values


def certificate(etas, q, checks, b):
    alphas = [1 - eta**2 + eta**2 * F(1, q**t) for eta, t in zip(etas, checks)]
    return math.prod(sorted(alphas, reverse=True)[:len(alphas) - b], start=F(1))


def load_planner(root, inventory):
    root = Path(root).resolve()
    receipt_path = root / "PLANNER_RECEIPT.json"
    receipt_sha = inventory.add(receipt_path, "exact planner receipt")
    receipt = read_json(receipt_path)
    for name in ("plans.json", "exact_certificates.json"):
        expected = receipt.get("artifacts", {}).get(name, {}).get("sha256")
        require(isinstance(expected, str), "Planner receipt does not bind " + name)
        inventory.add(root / name, "bound planner artifact", expected)
    raw_plan = (root / "plans.json").read_bytes()
    plan_file_sha = hashlib.sha256(raw_plan).hexdigest()
    obj, certs = read_json(root / "plans.json"), read_json(root / "exact_certificates.json")
    validation = receipt.get("group_reduction_validation", {})
    require(validation.get("mismatches") == 0 and validation.get("cases", 0) > 0
            and validation.get("allocations_compared", 0) > 0,
            "Planner lacks a completed exact enumeration-validation receipt")
    require((certs.get("k"), certs.get("b"), certs.get("R0"), certs.get("work_budget_gates"))
            == (K, 128, 1600, 204800), "Wrong frozen work-certificate parameters")
    target = exact(certs["target"])
    require(target == F(31, 50)**64, "Different predeclared target")
    plans, policy_output = {}, {}
    for item in obj["plans"]:
        name = item["name"]
        require(name in PLAN_NAMES and name not in plans, "Unexpected/duplicate plan")
        checks = expand(item["t_schedule"])
        etas = [F(x) for x in expand(item["eta_schedule"])]
        enc, q = item["encoding"], item["q"]
        pair = (F(1, 4), F(1, 3)) if enc == "sue" else (F(1, 10), F(1, 5))
        require(tuple(etas) == pair * 256, "Not the frozen matched-privacy input schedule")
        require(all(type(t) is int and t >= 1 for t in checks), "Invalid check schedule")
        require(item["typed_program"] == {"op": "outer_product_sum", "n": N, "r": R,
                                         "bilinear_gates_per_round": N * N * R}, "Wrong fixed program")
        require((item["n"], item["r"], item["k"]) == (N, R, K), "Wrong plan shape")
        work = certificate(etas, q, checks, 128)
        error_bound = min(F(1), sum((F(1, q**t) for t in checks), F(0)))
        planning_ns = item.get("planning_ns", item.get("planning_cost_ns"))
        require(type(planning_ns) is int and planning_ns >= 0, "Missing measured/declarative planning cost")
        if name in ("sue_opt17", "sue_rr17"):
            method = "optimal" if name == "sue_opt17" else "round_robin"
            result = certs["exact_planning"][method]
            require(q == 17 and enc == "sue", "Wrong mechanism plan")
            require(checks == result["t_schedule"] and sum(checks) == result["total_checks"], "Executed planner schedule mismatch")
            require(work == exact(result["bound"]) and work <= target, "Given schedule fails exact target certificate")
            require(planning_ns == result["planning_ns"] == receipt["planning_ns"][method], "Planning cost receipt mismatch")
            previous = result.get("previous_budget_bound")
            require(previous is not None and exact(previous) > target,
                    "Minimal-T result lacks a target-failing previous optimum certificate")
            if method == "round_robin":
                base, rem = divmod(sum(checks), K)
                require(checks == [base + (j < rem) for j in range(K)], "RR is not the frozen fixed-prefix rule")
        else:
            require(q == 65521 and checks == [2] * K, "Integrity plan is not q65521,t2")
            require(work == exact(certs[enc]["bound"]), "Integrity work certificate mismatch")
            require(error_bound == exact(certs[enc]["incorrect_increment_union_bound"]), "Output-integrity union bound mismatch")
        require(item["certificate"]["total_checks"] == sum(checks), "Declared total checks mismatch")
        digest = hashlib.sha256(json_bytes(item)).hexdigest()
        plans[name] = {"name": name, "encoding": enc, "q": q, "checks": checks,
                       "etas": etas, "digest": digest, "planning_ns": planning_ns}
        policy_output[name] = {"encoding": enc, "q": q, "total_checks": sum(checks),
                               "t_schedule": checks, "planning_ns": planning_ns,
                               "work_certificate": exact_summary(work), "work_target_met": work <= target,
                               "wrong_increment_union_bound": exact_summary(error_bound),
                               "honest_semantic_gates": K * N * N * R,
                               "work_budget_is_counterfactual": True}
    require(set(plans) == set(PLAN_NAMES), "The frozen four-plan set is incomplete")
    sue, oue = exact(certs["sue"]["bound"]), exact(certs["oue"]["bound"])
    with localcontext() as ctx:
        ctx.prec = 70
        work_orders = float((Decimal(oue.numerator) / Decimal(oue.denominator)).log10()
                            - (Decimal(sue.numerator) / Decimal(sue.denominator)).log10())
    output = {"plan_file_sha256": plan_file_sha, "receipt_sha256": receipt_sha,
              "source_sha256": receipt.get("source_sha256"), "host": receipt.get("hostname"),
              "planning_algorithm": receipt.get("planning_algorithm", "recorded exact planner"),
              "validation": validation, "target": exact_summary(target),
              "R0": 1600, "work_budget_gates": 204800, "b": 128, "k": K,
              "policies": policy_output,
              "integrity_mode_work_certificate_log10_oue_over_sue": work_orders,
              "integrity_mode_certificate_comparison_is_analytic_not_attack_rate": True,
              "minimality_scope": "given schedule certificate recomputed; external exact planner minimality receipt checked; optimizer not rerun"}
    return plans, output


def analytic_frequency_mse(records, enc):
    total = F(0)
    etas = (F(1, 4), F(1, 3)) if enc == "sue" else (F(1, 10), F(1, 5))
    low_count = 2 * (records // 4) + min(records % 4, 2)
    for eta, count in zip(etas, (low_count, records - low_count)):
        p = 1 - eta if enc == "sue" else F(1, 2)
        s0, s1 = eta * (1 - eta) / (p - eta)**2, p * (1 - p) / (p - eta)**2
        total += count * (((N - 1) * s0 + s1 + 1)**2 - 1)
    value = total / (N * N * records * records)
    # Independently derived closed forms for the equal-size strata in the declared prefixes.
    if records % 4 == 0:
        closed = F(47, 20 * records) if enc == "sue" else F(1925689, 1036800 * records)
        require(value == closed, "Analytical utility expressions disagree")
    return value


def source_key(source):
    return (source.get("source_sha256"), source.get("source_text_sha256"),
            source.get("category_sha256"), tuple(source.get("shape", ())))


def load_utility(roots, plans_info, inventory, harness_sha):
    require(len(roots) == 4, "Exactly four utility worker directories are required")
    data, truths, seen_reps = {}, {}, set()
    common_source, simulation = None, None
    for supplied in roots:
        root = Path(supplied).resolve()
        status = completed_directory(root, "utility", inventory, harness_sha)
        meta = read_json(root / "utility_metadata.json")
        require(meta["mode"] == "bulk_utility_simulation" and meta["security_fresh_episodes"] == 0,
                "Utility was relabelled as a security episode")
        require(meta["plan_file_sha256"] == plans_info["plan_file_sha256"], "Utility used different plans")
        require(tuple(meta["prefixes"]) == PREFIXES and meta["clipping"] is False,
                "Wrong utility prefix/clipping specification")
        require(meta["decoder_scale"] == SCALE and meta["utility_timer_excluded_from_service"] is True,
                "Wrong decoder or timing boundary")
        current_simulation = {name: meta[name] for name in
                              ("analysis_seed", "input_rng", "source_order", "cross_encoding_coupling",
                               "privacy_scope", "chunk_records", "top20_rule", "weighted_error_rule")}
        simulation = current_simulation if simulation is None else simulation
        require(current_simulation == simulation,
                "Utility workers did not share the declared simulation settings")
        key = source_key(meta["source"])
        common_source = key if common_source is None else common_source
        require(key == common_source, "Utility workers used different source data")
        reps = tuple(meta["replicate_ids"])
        require(len(reps) == 5 and len(set(reps)) == 5 and not (set(reps) & seen_reps),
                "Worker replicate IDs overlap or do not contain five units")
        require(status["result"]["completed_replicates"] == 5, "Worker did not complete five replicates")
        rows, completed = {}, {}
        for row in jsonl(root / "utility_rows.jsonl"):
            rep = row["replicate_id"]
            require(rep in reps, "Unplanned utility replicate")
            if row["kind"] == "utility":
                entry = (rep, row["records"], row["encoding"])
                require(entry not in rows and entry[1] in PREFIXES and entry[2] in ENCODINGS,
                        "Unexpected/duplicate utility row")
                require(row["clipped"] is False and row["prefix_population"] is True,
                        "Clipped estimate or wrong prefix population")
                rows[entry] = row
            elif row["kind"] == "replicate_complete":
                require(rep not in completed, "Duplicate utility completion record")
                completed[rep] = row
            else:
                raise ValueError("Unexpected utility record kind")
        require(set(completed) == set(reps), "A utility replicate lacks its completion artifact")
        require(len(rows) == 5 * len(PREFIXES) * 2, "Missing utility prefix/encoding rows")
        for rep in reps:
            record = completed[rep]
            file = child(root, record["estimate_file"])
            inventory.add(file, "full unclipped utility replicate", record["file_sha256"])
            with np.load(file, allow_pickle=False) as archive:
                require(int(archive["decoder_scale"]) == SCALE, "Replicate decoder scale mismatch")
                for count in PREFIXES:
                    truth = np.asarray(archive["truth_n_%d" % count])
                    require(truth.shape == (N, N) and truth.dtype == np.int64
                            and np.all(truth >= 0) and int(truth.sum()) == count, "Invalid prefix truth table")
                    if count in truths:
                        require(np.array_equal(truth, truths[count]), "Truth table differs between replicates")
                    else:
                        truths[count] = truth.copy()
                    top = np.argsort(-truth.reshape(-1), kind="stable")[:20]
                    for enc in ENCODINGS:
                        row = rows[(rep, count, enc)]
                        numerator = np.asarray(archive[enc + "_num_n_%d" % count])
                        estimate = np.asarray(archive[enc + "_estimate_n_%d" % count])
                        require(numerator.shape == (N, N) and numerator.dtype == np.int64,
                                "Invalid exact estimator numerator")
                        require(np.array_equal(estimate, numerator.astype(np.float64) / SCALE),
                                "Saved float estimator is not the uncut exact numerator/scale")
                        require(array_sha(numerator) == row["estimate_num_sha256"]
                                and array_sha(truth) == row["truth_sha256"], "Utility array hash mismatch")
                        require(row["true_top20_cell_indices"] == [int(x) for x in top],
                                "Top20 was not defined by the fixed true histogram")
                        error = estimate - truth
                        mse = float(np.mean(error**2))
                        top_mse = float(np.mean(error.reshape(-1)[top]**2))
                        weighted_mse = float(np.sum(truth.astype(np.float64) / count * error**2))
                        metrics = {"mse": mse, "frequency_mse": mse / count**2,
                                   "rmse": math.sqrt(mse), "frequency_rmse": math.sqrt(mse) / count,
                                   "mae": float(np.mean(np.abs(error))), "max_abs_error": float(np.max(np.abs(error))),
                                   "true_top20_rmse": math.sqrt(top_mse),
                                   "true_top20_frequency_rmse": math.sqrt(top_mse) / count,
                                   "true_top20_mae": float(np.mean(np.abs(error.reshape(-1)[top]))),
                                   "truth_weighted_rmse": math.sqrt(weighted_mse),
                                   "truth_weighted_frequency_rmse": math.sqrt(weighted_mse) / count,
                                   "total_mass_error": float(estimate.sum() - count)}
                        for name, value in metrics.items():
                            close(row[name], value, "utility " + name)
                        require(row["negative_cells"] == int(np.count_nonzero(numerator < 0)),
                                "Negative estimates were miscounted")
                        expected = analytic_frequency_mse(count, enc)
                        analytic = row["analytic_expected_mse"]
                        require(F(analytic["expected_frequency_mse_fraction"]) == expected,
                                "Recorded analytic E[MSE] disagrees with exact derivation")
                        close(analytic["sqrt_expected_frequency_mse"], math.sqrt(float(expected)),
                              "analytic sqrt(E[MSE])")
                        data[(rep, count, enc)] = dict(metrics, replicate_id=rep, records=count,
                                                      encoding=enc, negative_cells=row["negative_cells"],
                                                      analytic_frequency_mse=float(expected),
                                                      source_prefix_sha256=row["source_prefix_sha256"],
                                                      numerator_sha256=row["estimate_num_sha256"])
        seen_reps.update(reps)
    require(seen_reps == set(IDS) and len(data) == 20 * 4 * 2,
            "Full utility grid is not exactly IDs0..19 by four N by two encodings")
    for count in PREFIXES:
        hashes = {data[(rep, count, enc)]["source_prefix_sha256"] for rep in IDS for enc in ENCODINGS}
        require(len(hashes) == 1, "Utility source prefix identities differ")
    return data, truths, common_source, simulation


def load_tape_manifest(path, inventory, plan_file_sha, harness_sha):
    completed_directory(Path(path).parent, "acquire", inventory, harness_sha)
    digest = inventory.add(path, "common mechanism tape manifest")
    manifest = read_json(path)
    require(manifest["plan_file_sha256"] == plan_file_sha, "Tapes generated under other plans")
    require(manifest["mode"] == "client_only_tape_acquisition" and manifest["fresh_work_executions"] == 0,
            "The frozen mechanism experiment requires client-only acquisition, not additional live executions")
    result = {}
    for item in manifest["streams"]:
        sid = item["stream_id"]
        require(sid in IDS and sid not in result, "Bad tape stream IDs")
        require(item["source_row_start"] == sid * 20000 and item["source_row_stop"] == sid * 20000 + 1024,
                "Different predeclared source slices")
        require(item["fresh_work_executions"] == 0 and item["independent_os_input_realizations"] == 1,
                "Mechanism acquisition freshness was relabelled")
        tape_path = child(Path(path).parent, item["file"])
        inventory.add(tape_path, "common mechanism input tape", item["file_sha256"])
        with np.load(tape_path, allow_pickle=False) as archive:
            for label in ("sue_u", "sue_v", "oue_u", "oue_v"):
                value = archive[label]
                require(value.shape == (K, N, R) and value.dtype == np.uint8
                        and np.all(value <= 1) and array_sha(value) == item["array_sha256"][label],
                        "Acquired report tape array is invalid or does not match its recorded hash")
        result[sid] = item
    require(set(result) == set(IDS), "Twenty common mechanism tapes are incomplete")
    return result, digest, source_key(manifest["source"])


def validate_rounds(root, filename, streams, plans, names, replay, inventory):
    inventory.add(Path(root) / filename, "complete per-round protocol receipts")
    seen = {(name, sid): set() for name in names for sid in IDS}
    checks = dict.fromkeys(seen, 0)
    client = dict.fromkeys(seen, 0)
    for row in jsonl(Path(root) / filename):
        key = (row.get("plan"), row.get("stream_id"))
        require(key in seen and row.get("kind") == "round", "Unplanned round receipt")
        j = row["round"]
        require(type(j) is int and 0 <= j < K and j not in seen[key], "Duplicate/missing-order round")
        plan = plans[key[0]]
        require(row["accepted"] is True and row["reason"] == "accepted", "A measured honest candidate was rejected")
        require(row["q"] == plan["q"] and row["encoding"] == plan["encoding"]
                and F(row["eta"]) == plan["etas"][j], "Round field/input distribution mismatch")
        require(row["t_scheduled"] == row["checks_attempted"] == plan["checks"][j], "Executed check schedule mismatch")
        require(row["performance_replay"] is replay, "Replay/live receipt relabelled")
        require(row["committed_records"] == 2 * (j + 1) and row["bilinear_semantic_gates"] == N * N * R,
                "State progress/logical work receipt mismatch")
        require(row["candidate_bytes"] == N * N * 4, "Incomplete candidate interface")
        require(row["plan_fixed_ns"] <= row["candidate_frozen_ns"] <= row["first_challenge_ns"] <= row["decision_ns"],
                "Candidate-before-challenge ordering failed")
        require(len(row["challenge_sha256s"]) == plan["checks"][j], "Missing check receipts")
        require(row["total_ns"] > 0 and all(type(v) is int and v >= 0 for v in row["host_observed_phase_ns"].values()),
                "Invalid round timings")
        if not replay:
            require(row.get("fresh_work_execution") is True
                    and row["plan_fixed_ns"] <= row["input_generation_start_ns"] <= row["client_generation_end_ns"]
                    <= row["candidate_frozen_ns"], "Live fresh-input causal order failed")
            require(row["client_generation_ns"] == row["client_generation_end_ns"] - row["input_generation_start_ns"],
                    "Client span mismatch")
            client[key] += row["client_generation_ns"]
        else:
            require(row["input_generation_start_ns"] is None, "Replay was labelled fresh input")
        seen[key].add(j)
        checks[key] += row["checks_attempted"]
    for key, rounds in seen.items():
        require(rounds == set(range(K)), "A prescribed 512-round session is incomplete")
        require(checks[key] == streams[key]["checks_executed"] == sum(plans[key[0]]["checks"]),
                "Stream/round check counts disagree")
        if not replay:
            require(client[key] == streams[key]["client_generation_ns"], "Client spans do not sum to live stream decomposition")
    return {"sessions": len(seen), "rounds": len(seen) * K, "accepted_all": True,
            "checks": sum(checks.values()), "freshness_scope": "known-input replay" if replay else "OS-fresh recorded live order",
            "receipt_checks_do_not_provide_host_attestation": True}


def load_service(root, live, plans, plans_info, tapes, tapes_sha, inventory, harness_sha):
    root = Path(root).resolve()
    mode = "live_integrity" if live else "timing"
    completed_directory(root, mode, inventory, harness_sha)
    prefix = "integrity_live" if live else "timing"
    meta = read_json(root / (prefix + "_metadata.json"))
    require(meta["plan_file_sha256"] == plans_info["plan_file_sha256"], "Service used a different frozen plan file")
    names = ("sue_integrity65521", "oue_integrity65521") if live else ("sue_opt17", "sue_rr17")
    require({x["name"] for x in meta["plans"]} == set(names), "Unexpected service plan set")
    if not live:
        require(meta["tapes_manifest_sha256"] == tapes_sha, "Replay used a different input tape collection")
    for p in meta["plans"]:
        require(p["plan_sha256"] == plans[p["name"]]["digest"]
                and p["planning_cost_ns"] == plans[p["name"]]["planning_ns"], "Service plan/cost provenance mismatch")
    rows_path = root / (prefix + "_streams.jsonl")
    rows_sha = inventory.add(rows_path, "primary complete service sessions")
    streams = {}
    for row in jsonl(rows_path):
        key = row["plan"], row["stream_id"]
        require(key[0] in names and key[1] in IDS and key not in streams, "Unplanned/duplicate service session")
        plan = plans[key[0]]
        require(row["q"] == plan["q"] and row["encoding"] == plan["encoding"]
                and row["plan_sha256"] == plan["digest"], "Service session plan mismatch")
        require(row["accepted_rounds"] == K and row["committed_records"] == K * R,
                "Incomplete accepted stream")
        require(row["checks_executed"] == row["checks_scheduled"] == sum(plan["checks"]), "Wrong executed stream checks")
        require(row["planning_cost_ns_once_per_20_session_batch"] == plan["planning_ns"], "Wrong one-time planner cost")
        require(row["performance_replay"] is (not live), "Live/replay mismatch")
        if live:
            require(row["fresh_security_input_realizations"] == 1 and row["active_service_is_discontinuous_measure"] is True,
                    "Wrong live freshness/service-time interpretation")
            require(row["source_row_start"] == key[1] * 20000 and row["source_row_stop"] == key[1] * 20000 + 1024,
                    "Wrong live source block")
            require(row["source_slice_sha256"] == tapes[key[1]]["source_slice_sha256"],
                    "Live source slice differs from its predeclared common source block")
            require(row["observed_live_total_ns"] > 0 and row["active_service_ns"] > 0
                    and row["client_generation_ns"] >= row["client_os_ns"] >= 0
                    and row["observed_live_total_ns"] - row["client_generation_ns"] == row["active_service_ns"],
                    "Live time decomposition mismatch")
            raw = child(root, row["raw_file"])
            inventory.add(raw, "full live reports and decoded output", row["raw_file_sha256"])
            with np.load(raw, allow_pickle=False) as archive:
                for label in ("u", "v"):
                    value = archive[label]
                    require(value.shape == (K, N, R) and value.dtype == np.uint8
                            and np.all(value <= 1) and array_sha(value) == row[label + "_sha256"],
                            "Live report array identity mismatch")
                numerator, truth = archive["decoded_num"], archive["truth"]
                require(numerator.shape == (N, N) and numerator.dtype == np.int64
                        and array_sha(numerator) == row["decoded_num_sha256"]
                        and int(archive["decoded_scale"]) == SCALE, "Live output identity mismatch")
                require(truth.shape == (N, N) and np.all(truth >= 0) and int(truth.sum()) == 1024,
                        "Wrong live truth table")
                require(np.array_equal(archive["decoded_estimate"], numerator.astype(np.float64) / SCALE),
                        "Live estimate was clipped or altered")
        else:
            tape = tapes[key[1]]
            require(row["report_tape_sha256"] == tape["file_sha256"] and row["fresh_security_episodes_added"] == 0,
                    "Replay tape identity/freshness mismatch")
            require(row["service_total_ns"] > 0, "Invalid replay service cost")
            common, enc_cost = tape["client_common_ns"], tape["client_encoding_ns"][plan["encoding"]]
            require(row["client_common_ns"] == common and row["client_encoding_ns"] == enc_cost
                    and row["client_plus_service_charged_ns"] == common + enc_cost + row["service_total_ns"],
                    "Client-plus-replay charged total mismatch")
        streams[key] = row
    require(len(streams) == 40 and set(streams) == {(name, sid) for name in names for sid in IDS},
            "Required twenty sessions per service plan are incomplete")
    for sid in IDS:
        left, right = streams[(names[0], sid)], streams[(names[1], sid)]
        if live:
            require(left["source_slice_sha256"] == right["source_slice_sha256"], "Live comparison is not source-block paired")
        else:
            require(left["decoded_num_sha256"] == right["decoded_num_sha256"], "Same reports decoded differently by q17 policies")
    audit = validate_rounds(root, prefix + "_rounds.jsonl", streams, plans, names, not live, inventory)
    return {"streams": streams, "metadata": meta, "rows_sha256": rows_sha, "audit": audit,
            "mode": mode, "directory": str(root)}


def interval(values):
    return [float(x) for x in np.quantile(values, (0.025, 0.975))]


def utility_summary(data, indices):
    summary = []
    points = []
    for count in PREFIXES:
        entry = {"records": count, "encodings": {}}
        for enc in ENCODINGS:
            values = np.array([data[(rep, count, enc)]["frequency_mse"] for rep in IDS])
            top = np.array([data[(rep, count, enc)]["true_top20_frequency_rmse"] for rep in IDS])
            weighted = np.array([data[(rep, count, enc)]["truth_weighted_frequency_rmse"] for rep in IDS])
            boot_root = np.sqrt(values[indices].mean(axis=1))
            analytic = float(analytic_frequency_mse(count, enc))
            entry["encodings"][enc] = {"frequency_mean_mse": float(values.mean()),
                                       "frequency_root_mean_mse": float(math.sqrt(values.mean())),
                                       "root_mean_mse_nominal_95_interval": interval(boot_root),
                                       "analytic_expected_frequency_mse": analytic,
                                       "analytic_sqrt_expected_frequency_mse": math.sqrt(analytic),
                                       "empirical_mean_mse_over_analytic": float(values.mean() / analytic),
                                       "top20_frequency_rmse_median": float(np.median(top)),
                                       "top20_median_nominal_95_interval": interval(np.median(top[indices], axis=1)),
                                       "weighted_frequency_rmse_median": float(np.median(weighted)),
                                       "negative_cells_median": float(np.median([data[(r, count, enc)]["negative_cells"] for r in IDS]))}
            points.extend(data[(rep, count, enc)] for rep in IDS)
        sue = np.array([data[(rep, count, "sue")]["frequency_mse"] for rep in IDS])
        oue = np.array([data[(rep, count, "oue")]["frequency_mse"] for rep in IDS])
        sue_top = np.array([data[(rep, count, "sue")]["true_top20_frequency_rmse"] for rep in IDS])
        oue_top = np.array([data[(rep, count, "oue")]["true_top20_frequency_rmse"] for rep in IDS])
        require(np.all(oue > 0) and np.all(oue_top > 0), "A zero denominator prevents a declared utility ratio")
        entry["paired_contrasts"] = {"whole_table_root_mean_mse_sue_over_oue": float(math.sqrt(sue.mean()/oue.mean())),
                                      "whole_table_root_ratio_nominal_95_interval":
                                      interval(np.sqrt(sue[indices].mean(axis=1)/oue[indices].mean(axis=1))),
                                      "top20_median_paired_rmse_sue_over_oue": float(np.median(sue_top/oue_top)),
                                      "top20_paired_ratio_nominal_95_interval": interval(np.median((sue_top/oue_top)[indices], axis=1))}
        summary.append(entry)
    return summary, points


def mechanism_summary(service, plans, indices):
    rows = service["streams"]
    opt, rr = "sue_opt17", "sue_rr17"
    po, pr = plans[opt]["planning_ns"], plans[rr]["planning_ns"]
    points = []
    for sid in IDS:
        a, b = rows[(opt, sid)], rows[(rr, sid)]
        x, y = a["service_total_ns"], b["service_total_ns"]
        points.append({"stream_id": sid, "tape_sha256": a["report_tape_sha256"],
                       "opt_service_ns": x, "rr_service_ns": y,
                       "opt_amortized_ns": x + po/20, "rr_amortized_ns": y + pr/20,
                       "observed_service_ratio_rr_over_opt": y/x,
                       "planning_amortized_ratio_rr_over_opt": (y+pr/20)/(x+po/20),
                       "observed_service_difference_rr_minus_opt_ns": y-x,
                       "client_service_charged_ratio_rr_over_opt": b["client_plus_service_charged_ns"]/a["client_plus_service_charged_ns"],
                       "client_service_planning_amortized_ratio_rr_over_opt":
                       (b["client_plus_service_charged_ns"]+pr/20)/(a["client_plus_service_charged_ns"]+po/20)})
    ratios = np.array([p["planning_amortized_ratio_rr_over_opt"] for p in points])
    methods = {}
    for name in (opt, rr):
        samples = np.array([rows[(name, sid)]["service_total_ns"] for sid in IDS], dtype=np.int64)
        methods[name] = {"median_observed_service_ms": float(np.median(samples)/1e6),
                         "twenty_session_observed_sum_ms": int(samples.sum())/1e6,
                         "planning_once_ms": plans[name]["planning_ns"]/1e6,
                         "twenty_session_planning_inclusive_ms": (int(samples.sum())+plans[name]["planning_ns"])/1e6,
                         "twenty_session_charged_client_service_plus_once_planning_ms":
                         (sum(rows[(name,sid)]["client_plus_service_charged_ns"] for sid in IDS)+plans[name]["planning_ns"])/1e6,
                         "checks_per_stream": sum(plans[name]["checks"])}
    ratio_total = methods[rr]["twenty_session_planning_inclusive_ms"] / methods[opt]["twenty_session_planning_inclusive_ms"]
    return {"mode": "q17_resource_mechanism_performance_replay", "backend": service["metadata"]["backend"],
            "execution_environment": service["metadata"]["metadata"],
            "audit": service["audit"], "rows_sha256": service["rows_sha256"],
            "paired_points": points, "median_planning_amortized_ratio": float(np.median(ratios)),
            "nominal_95_percentile_interval": interval(np.median(ratios[indices], axis=1)),
            "twenty_session_planning_inclusive_ratio": ratio_total, "methods": methods,
            "twenty_session_charged_client_service_planning_ratio":
            methods[rr]["twenty_session_charged_client_service_plus_once_planning_ms"] /
            methods[opt]["twenty_session_charged_client_service_plus_once_planning_ms"],
            "planner_charging": "one shared search per twenty-session backend scenario; equal 1/20 allocation only for paired display",
            "not_an_output_integrity_or_cross_field_comparison": True}


def live_summary(service, plans, indices):
    rows = service["streams"]
    methods = {}
    for name in ("sue_integrity65521", "oue_integrity65521"):
        data = [rows[(name, sid)] for sid in IDS]
        methods[name] = {"median_observed_live_ms": float(np.median([r["observed_live_total_ns"] for r in data])/1e6),
                         "median_active_service_ms": float(np.median([r["active_service_ns"] for r in data])/1e6),
                         "median_client_ms": float(np.median([r["client_generation_ns"] for r in data])/1e6),
                         "twenty_session_observed_live_plus_once_planning_ms":
                         (sum(r["observed_live_total_ns"] for r in data)+plans[name]["planning_ns"])/1e6,
                         "twenty_session_active_service_plus_once_planning_ms":
                         (sum(r["active_service_ns"] for r in data)+plans[name]["planning_ns"])/1e6,
                         "checks_per_stream": sum(plans[name]["checks"]),
                         "wrong_increment_union_bound": exact_summary(F(K,65521**2)),
                         "fresh_live_sessions": 20}
    points = []
    for sid in IDS:
        a, b = rows[("sue_integrity65521",sid)], rows[("oue_integrity65521",sid)]
        points.append({"stream_id":sid, "source_slice_sha256":a["source_slice_sha256"],
                       "active_service_ratio_oue_over_sue":b["active_service_ns"]/a["active_service_ns"],
                       "observed_live_ratio_oue_over_sue":b["observed_live_total_ns"]/a["observed_live_total_ns"]})
    ratios=np.array([p["active_service_ratio_oue_over_sue"] for p in points])
    return {"mode":"q65521_fresh_live_integrity", "backend":service["metadata"]["backend"],
            "execution_environment":service["metadata"]["metadata"],
            "audit":service["audit"], "rows_sha256":service["rows_sha256"], "methods":methods,
            "paired_points":points, "pairing":"same source blocks; independent report coins between encodings",
            "median_source_paired_active_service_ratio_oue_over_sue":float(np.median(ratios)),
            "nominal_95_percentile_interval":interval(np.median(ratios[indices],axis=1)),
            "active_service_time_is_not_a_continuous_wall_interval":True,
            "privacy_noise_is_distinct_from_service_integrity_error":True}


def crosscheck_summaries(paths, outputs, inventory, harness_sha):
    by_sha = {value["rows_sha256"]:value for value in outputs}
    checked=[]
    for supplied in paths:
        path=Path(supplied).resolve()
        completed_directory(path.parent,"summarize",inventory,harness_sha)
        inventory.add(path,"independently generated harness summary")
        data=read_json(path)
        source=data.get("source_rows_sha256")
        require(source in by_sha,"Summary is not bound to one of these validated raw collections")
        target=by_sha[source]
        if target["mode"].startswith("q17"):
            require(len(data["contrasts"])==1,"Missing harness q17 contrast")
            c=data["contrasts"][0]
            close(c["median_amortized_ratio"],target["median_planning_amortized_ratio"],"independent paired median")
            close(c["twenty_session_planning_inclusive_ratio"],target["twenty_session_planning_inclusive_ratio"],"independent batch ratio")
            require({p["stream_id"] for p in c["paired_points"]}==set(IDS),"Summary omitted paired points")
        else:
            for method in data["methods"]:
                name=method["plan"]
                require(name in target["methods"],"Unknown live summary method")
                close(method["median_observed_live_ns"]/1e6,target["methods"][name]["median_observed_live_ms"],"live median")
                close(method["median_active_service_ns"]/1e6,target["methods"][name]["median_active_service_ms"],"active median")
        checked.append({"path":str(path),"source_rows_sha256":source,"checked":True})
    return checked


def render_figure(out, utility, utility_points, mechanism):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8,
                         "axes.titlesize":8.5,"axes.labelsize":8,"xtick.labelsize":7,
                         "ytick.labelsize":7,"legend.fontsize":6.5,"pdf.fonttype":42,
                         "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    colors={"sue":"#D55E00","oue":"#0072B2"}
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.7),layout="constrained",
                          gridspec_kw={"width_ratios":[1.06,.91,1.03]})
    ax=axes[0]
    for enc in ENCODINGS:
        xs=np.array([x["records"] for x in utility],dtype=float)
        ys=np.array([x["encodings"][enc]["frequency_root_mean_mse"] for x in utility])
        ci=np.array([x["encodings"][enc]["root_mean_mse_nominal_95_interval"] for x in utility])
        analytic=np.array([x["encodings"][enc]["analytic_sqrt_expected_frequency_mse"] for x in utility])
        for j,count in enumerate(PREFIXES):
            pts=sorted((p for p in utility_points if p["records"]==count and p["encoding"]==enc),key=lambda p:p["replicate_id"])
            jitter=np.exp(np.linspace(-.025,.025,20))
            ax.scatter(count*jitter,[p["frequency_rmse"] for p in pts],s=5,color=colors[enc],alpha=.28,linewidths=0,zorder=2)
        ax.vlines(xs,ci[:,0],ci[:,1],color=colors[enc],lw=.8,zorder=4)
        for x,(lo,hi) in zip(xs,ci):
            ax.hlines([lo,hi],x/1.025,x*1.025,color=colors[enc],lw=.8,zorder=4)
        ax.plot(xs,ys,"o-",ms=2.8,lw=1.1,color=colors[enc],zorder=4)
        ax.plot(xs,analytic,ls="--",lw=.9,color=colors[enc],zorder=3)
    ax.set(xscale="log",yscale="log",xlabel="Records N",ylabel="Frequency root mean MSE",title="(a) Whole table")
    ax.set_xticks(PREFIXES,["16k","66k","262k","581k"])
    ax.tick_params(axis="x",which="minor",bottom=False)
    ax.grid(axis="y",alpha=.22,linewidth=.5)
    ax.legend(handles=[Line2D([],[],color=colors[e],marker="o",ms=3,lw=1,label=e.upper()) for e in ENCODINGS]
                      +[Line2D([],[],color=".35",ls="--",lw=1,label="Analytic")],loc="upper right",frameon=False,
              handlelength=1.5,borderpad=.2,labelspacing=.2)
    ax=axes[1]
    full=utility[-1]
    by_rep={(p["replicate_id"],p["encoding"]):p for p in utility_points if p["records"]==PREFIXES[-1]}
    offsets=np.linspace(-.055,.055,20)
    for rep in IDS:
        values=[by_rep[(rep,e)]["true_top20_frequency_rmse"] for e in ENCODINGS]
        ax.plot([offsets[rep],1+offsets[rep]],values,color=".58",alpha=.4,lw=.45,zorder=1)
        for x,e,y in zip((offsets[rep],1+offsets[rep]),ENCODINGS,values):
            ax.scatter([x],[y],s=11,color=colors[e],alpha=.8,linewidths=0,zorder=3)
    for x,enc in enumerate(ENCODINGS):
        values=full["encodings"][enc]
        mid=values["top20_frequency_rmse_median"]
        lo,hi=values["top20_median_nominal_95_interval"]
        ax.vlines([x],[lo],[hi],color="black",lw=1,zorder=4)
        ax.hlines([lo,hi],x-.04,x+.04,color="black",lw=1,zorder=4)
        ax.plot([x],[mid],marker="_",ms=15,mew=1.5,color="black",zorder=4)
    ax.set_xticks([0,1],["SUE","OUE"])
    ax.set(xlim=(-.28,1.28),ylabel="Top-20 frequency RMSE",title="(b) Frequent cells, N=581k")
    ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2),useMathText=True)
    ax.grid(axis="y",alpha=.22,lw=.5)
    ax.legend(handles=[Line2D([],[],color=".58",lw=.7,label="Replicate pair"),
                       Line2D([],[],color="black",marker="_",ms=9,lw=1,label="Median / interval")],
              loc="best",frameon=False,handlelength=1.5,borderpad=.2,labelspacing=.2)
    ax=axes[2]
    points=mechanism["paired_points"]
    values=np.array([p["planning_amortized_ratio_rr_over_opt"] for p in points])
    lo,hi=mechanism["nominal_95_percentile_interval"]
    mid=mechanism["median_planning_amortized_ratio"]
    ax.axhline(1,color=".45",ls=":",lw=.9,zorder=1)
    ax.axhspan(lo,hi,color="#009E73",alpha=.16,zorder=1)
    ax.axhline(mid,color="#007A58",lw=1,zorder=2)
    ax.scatter(np.arange(1,21),values,color="#007A58",s=13,linewidths=0,zorder=3)
    ax.set(xlim=(.3,20.7),xlabel="Paired stream",ylabel="RR / OPT time",title="(c) CPU, q=17")
    ax.set_xticks([1,5,10,15,20])
    lower=min(1,float(values.min()),lo)
    upper=max(1,float(values.max()),hi)
    padding=max(.025,(upper-lower)*.16)
    ax.set_ylim(max(0,lower-padding),upper+padding)
    ax.grid(axis="y",alpha=.22,lw=.5)
    ax.legend(handles=[Line2D([],[],color="#007A58",marker="o",ms=3,lw=0,label="20 tape pairs"),
                       Line2D([],[],color="#007A58",lw=1.1,label="Median / interval")],
              loc="best",frameon=False,handlelength=1.5,borderpad=.2,labelspacing=.2)
    fig.savefig(out/"application_results.pdf")
    fig.savefig(out/"application_results.png",dpi=300)
    plt.close(fig)
    return {"width_inches":7.1,"height_inches":2.7,"panels":3,"png_dpi":300,
            "matplotlib_version":matplotlib.__version__,
            "global_legend":False,"all_twenty_points_preserved":True,
            "a_statistic":"sqrt(mean replicate frequency MSE); individual points are replicate frequency RMSE",
            "b_statistic":"twenty paired true-top20 frequency RMSE values; median and nominal interval",
            "c_statistic":"twenty CPU same-tape RR/OPT service ratios with planner/20 allocated to each session"}


def narrative(utility, cpu, planner):
    last=utility[-1]
    whole=last["paired_contrasts"]["whole_table_root_mean_mse_sue_over_oue"]
    top=last["paired_contrasts"]["top20_median_paired_rmse_sue_over_oue"]
    speed=cpu["mechanism"]["median_planning_amortized_ratio"]
    interval_value=cpu["mechanism"]["nominal_95_percentile_interval"]
    return {"whole_table": "At the full source prefix, the observed SUE/OUE whole-table frequency root-mean-MSE ratio is %.6g; values above one favor OUE on this metric."%whole,
            "frequent_cells": "The median paired SUE/OUE true-top20 frequency-RMSE ratio is %.6g; this separate metric uses cells selected by the true histogram before inspecting noisy estimates."%top,
            "cpu_resource_cost": "The CPU median planning-amortized RR/OPT replay ratio is %.6g with nominal 95%% percentile interval [%.6g, %.6g]. Values above one favor OPT; parity or a reversal is retained."%(speed,*interval_value),
            "certificate": "The two q65521 work certificates differ by %.6g log10 units (OUE minus SUE). This is an analytical upper-bound comparison under the stated work hypothesis, not a measured attack probability or an output-integrity bound."%planner["integrity_mode_work_certificate_log10_oue_over_sue"],
            "scope": ["No estimator was clipped; all 1600 cells and unfavorable outcomes are retained.",
                      "Utility replicates are independently simulated randomizations of fixed prefixes, not new datasets or fresh security episodes.",
                      "Q17 uses known-input performance replays; q65521 service runs use new OS report coins after each fixed plan.",
                      "Bootstrap intervals are nominal and conditional on these input units and observed shared-hardware schedules.",
                      "The fixed trusted evaluator and recorded hashes do not provide hostile-host isolation or remote attestation.",
                      "No CPU/GPU pooling or cross-field same-integrity speedup is reported."]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planner-dir",required=True)
    ap.add_argument("--utility-dirs",nargs=4,required=True)
    ap.add_argument("--tapes-manifest",required=True)
    ap.add_argument("--cpu-replay",required=True)
    ap.add_argument("--cpu-live",required=True)
    ap.add_argument("--gpu-replay")
    ap.add_argument("--gpu-live")
    ap.add_argument("--gpu-omission-reason")
    ap.add_argument("--summary-jsons",nargs="*",default=[])
    ap.add_argument("--expected-harness-sha256",default="12d89f818443d11a319be5e636669b87361db6cf1bd0c3efae375ad491dccf17")
    ap.add_argument("--resamples",type=int,default=10000)
    ap.add_argument("--analysis-seed",type=int,default=20261110)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    require(args.resamples>=1000,"At least 1000 bootstrap resamples are required")
    require(bool(args.gpu_replay)==bool(args.gpu_live),"Supply both GPU collections or explicitly omit the GPU cell")
    require(bool(args.gpu_replay) or bool(args.gpu_omission_reason),
            "A CPU-only analysis requires an explicit GPU omission reason; preserve any failed raw collection")
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    inventory=Inventory()
    started=time.monotonic()
    try:
        plans,planner=load_planner(args.planner_dir,inventory)
        raw_utility,truths,source,simulation=load_utility(args.utility_dirs,planner,inventory,args.expected_harness_sha256)
        tapes,tapes_sha,tape_source=load_tape_manifest(args.tapes_manifest,inventory,planner["plan_file_sha256"],args.expected_harness_sha256)
        require(tape_source==source,"Timing tapes and utility use different categorical sources")
        idx=np.random.default_rng(args.analysis_seed).integers(0,20,size=(args.resamples,20))
        utility,utility_points=utility_summary(raw_utility,idx)
        services,summary_outputs={},[]
        for backend,replay_path,live_path in (("cpu",args.cpu_replay,args.cpu_live),
                                             ("gpu",args.gpu_replay,args.gpu_live)):
            if replay_path is None:
                continue
            replay=load_service(replay_path,False,plans,planner,tapes,tapes_sha,inventory,args.expected_harness_sha256)
            live=load_service(live_path,True,plans,planner,tapes,tapes_sha,inventory,args.expected_harness_sha256)
            require(replay["metadata"]["backend"]["kind"]==backend
                    and live["metadata"]["backend"]["kind"]==backend,"Backend label mismatch")
            require(source_key(live["metadata"]["source"])==source,"Live integrity used a different source")
            services[backend]={"mechanism":mechanism_summary(replay,plans,idx),
                               "integrity":live_summary(live,plans,idx)}
            summary_outputs.extend(services[backend].values())
        checked=crosscheck_summaries(args.summary_jsons,summary_outputs,inventory,args.expected_harness_sha256)
        figure=render_figure(out,utility,utility_points,services["cpu"]["mechanism"])
        descriptions=narrative(utility,services["cpu"],planner)
        settings={"n":N,"r":R,"rounds_per_stream":K,"utility_prefixes":list(PREFIXES),
                  "independent_units":20,"bootstrap_resamples":args.resamples,"analysis_seed":args.analysis_seed,
                  "bootstrap":"paired percentile; same resampled IDs for encodings, prefixes and paired contrasts",
                  "utility_simulation_settings":simulation,
                  "gpu_collection_status":"included" if args.gpu_replay else "omitted: "+args.gpu_omission_reason,
                  "root_mean_mse_not_mean_rmse":True,"unclipped_raw_inputs_retained":True,
                  "no_existing_experiment_statistics_recomputed":True}
        complete={"schema":"connected-application-analysis-v1","status":"validated_complete",
                  "settings":settings,"planner":planner,"utility":utility,"timing":services,
                  "descriptions":descriptions,"figure":figure,"summary_crosschecks":checked,
                  "source_identity":source,"inputs":inventory.records(),"analysis_script_sha256":sha_file(__file__)}
        complete["analysis_environment"]={"hostname":platform.node(),"python":sys.version,
                                          "platform":platform.platform(),"numpy":np.__version__}
        digest={"schema":"connected-numbers-digest-v1","settings":settings,"planner":planner,
                "full_source_utility":utility[-1],"all_prefix_utility":utility,
                "cpu_mechanism":{k:v for k,v in services["cpu"]["mechanism"].items() if k!="paired_points"},
                "live_integrity_tables":{backend:value["integrity"]["methods"] for backend,value in services.items()},
                "gpu_mechanism":None if "gpu" not in services else
                {k:v for k,v in services["gpu"]["mechanism"].items() if k!="paired_points"},
                "descriptions":descriptions}
        pairs={"utility_points":utility_points,
               "mechanism":{b:v["mechanism"]["paired_points"] for b,v in services.items()},
               "integrity_source_pairs":{b:v["integrity"]["paired_points"] for b,v in services.items()}}
        write_json(out/"connected_complete_results.json",complete)
        write_json(out/"connected_numbers_digest.json",digest)
        write_json(out/"connected_paired_points.json",pairs)
        write_json(out/"analysis_status.json",{"status":"completed","elapsed_seconds":time.monotonic()-started,
                   "input_count":len(inventory.items),"analysis_script_sha256":sha_file(__file__),
                   "artifacts":{p.name:{"bytes":p.stat().st_size,"sha256":sha_file(p)}
                                for p in out.iterdir() if p.is_file() and p.name!="analysis_status.json"}})
        print(json.dumps({"status":"completed","out":str(out),"inputs":len(inventory.items),
                          "gpu_included":"gpu" in services,"figure_inches":[7.1,2.7]}))
        return 0
    except Exception as exc:
        write_json(out/"analysis_status.json",{"status":"incomplete","error":type(exc).__name__+": "+str(exc),
                   "elapsed_seconds":time.monotonic()-started,"inputs_checked":inventory.records()})
        raise


if __name__=="__main__":
    raise SystemExit(main())
