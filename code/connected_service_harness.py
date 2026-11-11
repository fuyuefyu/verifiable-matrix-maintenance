#!/usr/bin/env python3
"""Pro-authored connected application harness; execute only on authorized workers.

This file was authored without running a benchmark or scientific calculation.
Commands: prepare, validate, acquire, timing, live_integrity, utility, summarize.
Live OS report randomization and performance replay have distinct receipts.
Utility is a separate, explicitly seeded bulk simulation, never a security run.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile
from fractions import Fraction

# Set before importing NumPy/BLAS. Never use a reserved environment variable.
for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "4"
import numpy as np

N = 40
R = 2
K = 512
RECORDS = R * K
SCALE = 144
SOURCE_ZIP_SHA = "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d"
SOURCE_TEXT_SHA = "0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945"
SCHEMA = "connected-cooccurrence-v1"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json_bytes({"shape": list(value.shape), "dtype": value.dtype.str}))
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False).encode("utf-8") + b"\n")


def emit(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


class Deadline:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or not 10 < seconds <= 1800:
            raise ValueError("max-seconds must be in (10,1800]")
        self.start = time.monotonic()
        self.stop = self.start + seconds - 5

    def check(self):
        if time.monotonic() >= self.stop:
            raise TimeoutError("Cooperative deadline reached; external hard deadline also required")


def new_out(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def metadata():
    result = {"schema": SCHEMA, "python": sys.version, "numpy": np.__version__,
              "hostname": platform.node(), "platform": platform.platform(),
              "processor": platform.processor(), "machine": platform.machine(),
              "thread_env": {key: os.environ.get(key) for key in
                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "script_sha256": sha_file(__file__), "started_unix_ns": time.time_ns()}
    try:
        from threadpoolctl import threadpool_info
        result["threadpools"] = threadpool_info()
    except ImportError:
        result["threadpools"] = "threadpoolctl unavailable"
    return result


def fraction(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("Exact probabilities must be Fraction strings or integer values, not floats")
    return Fraction(value)


def expand(values, label):
    if not isinstance(values, list) or len(values) not in (8, K):
        raise ValueError(label + " must contain 8 or 512 entries")
    return tuple(values * (K // 8) if len(values) == 8 else values)


@dataclasses.dataclass(frozen=True)
class Plan:
    name: str
    purpose: str
    encoding: str
    q: int
    t: tuple
    eta: tuple
    planning_cost_ns: int
    planning_cost_source: str
    declared_certificate: dict
    digest: str


def load_plans(path):
    plan_path = Path(path).resolve()
    raw = plan_path.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj.get("plans"), list) or not obj["plans"]:
        raise ValueError("Plan file must contain a nonempty plans list")
    receipt_path = plan_path.parent / "PLANNER_RECEIPT.json"
    if not receipt_path.is_file():
        raise ValueError("Keep the exact PLANNER_RECEIPT.json beside plans.json")
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    artifact = receipt.get("artifacts", {}).get("plans.json", {})
    if artifact.get("sha256") != sha_bytes(raw):
        raise ValueError("PLANNER_RECEIPT.json does not bind this exact plans.json")
    provenance = "PLANNER_RECEIPT.json sha256=" + sha_bytes(receipt_raw)
    plans = []
    seen = set()
    for item in obj["plans"]:
        if item.get("n", N) != N or item.get("r", R) != R or item.get("k", K) != K:
            raise ValueError("This predeclared experiment requires n=40,r=2,k=512")
        name = item["name"]
        if (not isinstance(name, str) or not name or name in seen or
                not name[0].isalnum() or any(not (c.isalnum() or c in "_.-") for c in name)):
            raise ValueError("Plan names must be nonempty and unique")
        seen.add(name)
        enc, q = item["encoding"], item["q"]
        if enc not in ("sue", "oue") or type(q) is not int or q not in (17, 65521):
            raise ValueError("Invalid encoding/field")
        ts = expand(item["t_schedule"], "t_schedule")
        if any(type(t) is not int or not 1 <= t <= 64 for t in ts):
            raise ValueError("Each scheduled check count must be an integer in [1,64]")
        es = tuple(fraction(x) for x in expand(item["eta_schedule"], "eta_schedule"))
        allowed = {Fraction(1, 4), Fraction(1, 3)} if enc == "sue" else {Fraction(1, 10), Fraction(1, 5)}
        if any(e not in allowed for e in es):
            raise ValueError("eta_schedule must use the predeclared actual false-bit probabilities")
        purpose = "mechanism" if q == 17 else "integrity"
        requested_purpose = item.get("purpose", purpose)
        if requested_purpose not in (purpose, "resource_certificate" if q == 17 else purpose):
            raise ValueError("q=17 mechanism and q=65521 integrity purposes must not be relabelled")
        if q == 65521 and min(ts) < 2:
            raise ValueError("The integrity configuration requires at least two checks per round")
        declared_program = item.get("typed_program")
        expected_program = {"op": "outer_product_sum", "n": N, "r": R,
                            "bilinear_gates_per_round": N * N * R}
        if declared_program != expected_program:
            raise ValueError("Only the declared fixed outer_product_sum n40 r2 program is admitted here")
        cost = item.get("planning_cost_ns", item.get("planner_cost_ns", item.get("planning_ns", 0)))
        if type(cost) is not int or cost < 0:
            raise ValueError("planning_cost_ns must be a nonnegative integer")
        method = {"sue_opt17": "optimal", "sue_rr17": "round_robin"}.get(name)
        if method is not None:
            if receipt.get("planning_ns", {}).get(method) != cost:
                raise ValueError("Plan's planning_ns differs from the bound planner receipt")
            source = provenance + "; measured planning_ns[" + method + "]"
        elif q == 65521 and cost == 0:
            source = provenance + "; fixed t rule, no separately timed planning search; instance/read cost remains timed"
        else:
            source = provenance + "; " + str(item.get("planning_cost_source", "declared planning charge, not independently measured by this harness"))
        cert = item.get("certificate", {})
        if not isinstance(cert, dict):
            raise ValueError("certificate must be declarative JSON data")
        if cert.get("total_checks", sum(ts)) != sum(ts):
            raise ValueError("Declared certificate total_checks does not match the executed schedule")
        # No arbitrary code, callable, module, native kernel, or input-dependent dispatch is loaded.
        plans.append(Plan(name, purpose, enc, q, tuple(ts), es, cost, str(source), cert,
                          sha_bytes(json_bytes(item))))
    by_enc = {}
    for plan in plans:
        if plan.encoding in by_enc and plan.eta != by_enc[plan.encoding]:
            raise ValueError("All policies for an encoding must share the same report distribution")
        by_enc[plan.encoding] = plan.eta
    if "sue" in by_enc and "oue" in by_enc:
        for a, b in zip(by_enc["sue"], by_enc["oue"]):
            if b != a * a / (a * a + (1 - a) * (1 - a)):
                raise ValueError("SUE and OUE schedules do not have matched marginal privacy")
    return plans, sha_bytes(raw), obj


def choose_plans(plans, names):
    if not names:
        return plans
    requested = [x.strip() for x in names.split(",") if x.strip()]
    by_name = {p.name: p for p in plans}
    if not requested or len(requested) != len(set(requested)) or any(x not in by_name for x in requested):
        raise ValueError("--plan-names must name existing unique plans")
    return [by_name[name] for name in requested]


def decoder_coefficients(enc, eta):
    p = 1 - eta if enc == "sue" else Fraction(1, 2)
    gap = p - eta
    values = (Fraction(SCALE, 1) / gap**2,
              -Fraction(SCALE, 1) * eta / gap**2,
              Fraction(SCALE, 1) * eta**2 / gap**2)
    if any(value.denominator != 1 for value in values):
        raise ValueError("Chosen decoder does not fit the fixed exact integer scale")
    return tuple(int(value) for value in values)


def decoded_delta(delta, uh, vh, enc, eta):
    a, b, c = decoder_coefficients(enc, eta)
    mu = uh.sum(axis=1, dtype=np.int64)
    mv = vh.sum(axis=1, dtype=np.int64)
    return a * delta + b * (mu[:, None] + mv[None, :]) + c * uh.shape[1]


def os_uniform(modulus, count):
    """Exact rejection sampling; consumption happens at this call, not at plan construction."""
    if not 2 <= modulus <= 65535:
        raise ValueError("Unsupported OS uniform modulus")
    if modulus <= 256:
        dtype, size, space = np.dtype("u1"), 1, 256
    else:
        dtype, size, space = np.dtype("<u2"), 2, 65536
    limit = (space // modulus) * modulus
    result = np.empty(count, dtype=np.uint32)
    used = 0
    while used < count:
        request = max(64, min(1 << 20, count - used + 32))
        draws = np.frombuffer(os.urandom(size * request), dtype=dtype)
        good = draws[draws.astype(np.uint32) < limit].astype(np.uint32) % modulus
        take = min(good.size, count - used)
        result[used:used + take] = good[:take]
        used += take
    return result


def encode_thresholds60(draws, categories, true_threshold, false_threshold):
    """Vectorized exact integer thresholds; independent coins exist for every bit."""
    cats = np.asarray(categories, dtype=np.int64)
    true_bit = np.arange(N)[:, None] == cats[None, :]
    thresholds = np.where(true_bit, np.asarray(true_threshold, dtype=np.uint8)[None, :],
                          np.asarray(false_threshold, dtype=np.uint8)[None, :])
    return np.ascontiguousarray(draws < thresholds, dtype=np.uint8)


def encode_from_uniform60(draws, categories, enc, etas):
    """Small live batches use exact Fraction-derived thresholds; bulk uses their fixed table."""
    false_threshold = np.array([int(e * 60) for e in etas], dtype=np.uint8)
    ps = [1 - e if enc == "sue" else Fraction(1, 2) for e in etas]
    true_threshold = np.array([int(p * 60) for p in ps], dtype=np.uint8)
    if any((e * 60).denominator != 1 for e in etas) or any((p * 60).denominator != 1 for p in ps):
        raise ValueError("Uniform-60 sampler cannot exactly realize a requested probability")
    return encode_thresholds60(draws, categories, true_threshold, false_threshold)


class Backend:
    def __init__(self, name, device):
        self.name = name
        self.xp = np
        self.gpu = name == "gpu"
        self.details = {"kind": name, "device": device}
        if self.gpu:
            import cupy as cp
            cp.cuda.Device(device).use()
            self.xp = cp
            props = cp.cuda.runtime.getDeviceProperties(device)
            devname = props["name"]
            self.details.update(device_name=devname.decode() if isinstance(devname, bytes) else str(devname),
                                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                cuda_driver=cp.cuda.runtime.driverGetVersion())

    def sync(self):
        if self.gpu:
            self.xp.cuda.runtime.deviceSynchronize()

    def array(self, value):
        return self.xp.array(value, dtype=self.xp.float64, order="C", copy=True)

    def host(self, value):
        return self.xp.asnumpy(value) if self.gpu else np.asarray(value)

    def mod(self, value, q):
        return self.xp.remainder(value.astype(self.xp.int64), q).astype(self.xp.float64)

    def mm(self, left, right, q):
        if int(left.shape[1]) * (q - 1)**2 >= 2**53:
            raise ValueError("FP64 exact-integer accumulation bound exceeded")
        return self.mod(left @ right, q)

    def truth(self, value):
        return bool(value.item()) if self.gpu else bool(value)

    def canonical(self, value, q):
        encoded = value.astype(self.xp.uint32)
        if not self.truth(self.xp.all((value >= 0) & (value < q) & (value == encoded))):
            raise ValueError("Noncanonical finite-field value")
        result = np.array(self.host(encoded), dtype="<u4", order="C", copy=True)
        result.flags.writeable = False
        return result


class Runtime:
    """Closed trusted evaluator: fixed outer products, same reference, atomic acceptance.

    This local control is not remote attestation or hostile-host isolation.
    Semantic bilinear gates: n*n*r per honest round; GPU instructions are not counted.
    """
    def __init__(self, ctx, plan, stream_id, test_mode=False):
        self.ctx, self.plan, self.stream_id = ctx, plan, stream_id
        self.test_mode = test_mode
        self.plan_fixed_ns = time.perf_counter_ns()
        self.c = ctx.array(np.zeros((N, N), dtype=np.uint32))
        self.host_c = np.zeros((N, N), dtype="<u4")
        self.host_c.flags.writeable = False
        self.decoded_num = np.zeros((N, N), dtype=np.int64)
        self.accepted_rounds = 0
        self.committed_records = 0
        self.stopped = False

    def fingerprint(self):
        return (array_sha(self.host_c), array_sha(self.decoded_num),
                self.accepted_rounds, self.committed_records)

    def process(self, j, uh, vh, deadline, input_generated_after_plan_ns=None,
                tamper_entry=None, test_vectors=None, mutate_original=False):
        if self.stopped or j != self.accepted_rounds:
            raise ValueError("Stopped sequence or out-of-order round")
        if any(x is not None for x in (tamper_entry, test_vectors)) or mutate_original:
            if not self.test_mode:
                raise ValueError("Test hooks are forbidden outside the validation command")
        if uh.shape != (N, R) or vh.shape != (N, R):
            raise ValueError("Incorrect factor shape")
        if not np.isin(uh, (0, 1)).all() or not np.isin(vh, (0, 1)).all():
            raise ValueError("Reports must be binary")
        if input_generated_after_plan_ns is not None and input_generated_after_plan_ns < self.plan_fixed_ns:
            raise ValueError("Fresh input was generated before this plan was fixed")
        deadline.check()
        q, ctx = self.plan.q, self.ctx
        phases = {}

        def phase(name, action):
            begin = time.perf_counter_ns()
            value = action()
            phases[name] = phases.get(name, 0) + time.perf_counter_ns() - begin
            return value

        start = time.perf_counter_ns()
        u, v = phase("report_transfer", lambda: (ctx.array(uh), ctx.array(vh)))
        # Exactly two fixed outer-product primitives; no bit-indexed table/branch algorithm.
        delta_product = phase("fixed_bilinear_program", lambda:
                              u[:, 0:1] * v[:, 0:1].T + u[:, 1:2] * v[:, 1:2].T)
        proposed = phase("candidate_construct", lambda: ctx.mod(self.c + delta_product, q))
        if tamper_entry is not None:
            a, b, amount = tamper_entry
            proposed[a, b] = (proposed[a, b] + amount) % q
        frozen = phase("candidate_freeze", lambda: ctx.canonical(proposed, q))
        context = {"plan": self.plan.digest, "stream": self.stream_id, "round": j, "q": q}
        digest = phase("candidate_hash", lambda: sha_bytes(json_bytes(context) + frozen.tobytes()))
        frozen_ns = time.perf_counter_ns()
        received = phase("candidate_receive", lambda: ctx.array(frozen))
        if mutate_original:
            proposed[0, 0] = (proposed[0, 0] + 1) % q
        lifted = phase("increment_lift", lambda:
                       np.remainder(frozen.astype(np.int64) - self.host_c.astype(np.int64), q))
        accepted = bool(np.all((lifted >= 0) & (lifted <= R)))
        reason = "accepted" if accepted else "increment_range"
        first_challenge_ns = None
        checks_attempted = 0
        challenge_hashes = []
        difference = phase("checker_difference", lambda: ctx.mod(received - self.c, q))
        if accepted:
            for h in range(self.plan.t[j]):
                deadline.check()
                stamp = time.perf_counter_ns()
                if first_challenge_ns is None:
                    first_challenge_ns = stamp
                if test_vectors is None:
                    zh = phase("challenge_os", lambda: os_uniform(q, N).reshape(N, 1))
                else:
                    zh = np.array(test_vectors[h], dtype=np.uint32).reshape(N, 1)
                z = phase("challenge_transfer", lambda: ctx.array(zh))

                def verify_one():
                    lhs = ctx.mm(difference, z, q)
                    rhs = ctx.mm(u, ctx.mm(v.T, z, q), q)
                    return ctx.truth(ctx.xp.all(lhs == rhs))

                ok = phase("sequential_verification", verify_one)
                checks_attempted += 1
                # A small challenge digest records freshness provenance; charged consistently.
                challenge_hashes.append(phase("challenge_receipt_hash", lambda: array_sha(zh)))
                if not ok:
                    accepted, reason = False, "freivalds_reject"
                    break
        decision_ns = time.perf_counter_ns()
        if accepted:
            increment_num = phase("integer_decoder", lambda:
                                  decoded_delta(lifted, uh, vh, self.plan.encoding, self.plan.eta[j]))

            def advance():
                # All candidate-dependent verification is complete before any of these writes.
                self.c = received
                self.host_c = frozen
                self.decoded_num += increment_num
                self.accepted_rounds += 1
                self.committed_records += R

            phase("atomic_state_advance", advance)
        else:
            self.stopped = True
        ctx.sync()
        end = time.perf_counter_ns()
        if first_challenge_ns is not None and not frozen_ns <= first_challenge_ns <= decision_ns:
            raise AssertionError("Candidate/challenge causal order broken")
        return {"kind": "round", "plan": self.plan.name, "stream_id": self.stream_id,
                "round": j, "q": q, "encoding": self.plan.encoding,
                "eta": str(self.plan.eta[j]), "t_scheduled": self.plan.t[j],
                "checks_attempted": checks_attempted, "accepted": accepted, "reason": reason,
                "total_ns": end - start, "host_observed_phase_ns": phases,
                "phase_semantics": "host calls and necessary waits; not synchronized component medians",
                "plan_fixed_ns": self.plan_fixed_ns,
                "input_generation_start_ns": input_generated_after_plan_ns,
                "candidate_frozen_ns": frozen_ns, "first_challenge_ns": first_challenge_ns,
                "decision_ns": decision_ns, "candidate_sha256": digest,
                "challenge_sha256s": challenge_hashes, "candidate_bytes": int(frozen.nbytes),
                "bilinear_semantic_gates": N * N * R,
                "committed_records": self.committed_records}


def prepare(args, out, deadline):
    source = Path(args.source).resolve()
    source_hash = sha_file(source)
    if args.source_sha256 and source_hash != args.source_sha256:
        raise ValueError("Source archive hash does not match the supplied identity")
    deadline.check()
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            if args.member not in archive.namelist():
                raise ValueError("Requested exact archive member is absent")
            with archive.open(args.member) as member:
                with gzip.GzipFile(fileobj=member) as decoded:
                    raw = decoded.read()
    elif source.suffix == ".gz":
        with gzip.open(source, "rb") as stream:
            raw = stream.read()
    else:
        raw = source.read_bytes()
    text_hash = sha_bytes(raw)
    if args.text_sha256 and text_hash != args.text_sha256:
        raise ValueError("Uncompressed source text hash mismatch")
    deadline.check()
    data = np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.int32)
    if data.shape != (581012, 55):
        raise ValueError("Expected canonical Covertype 581012 by 55 integer data")
    elevation = data[:, 0]
    soil = data[:, 14:54]
    if int(elevation.min()) < 0 or int(elevation.max()) >= 4000:
        raise ValueError("Elevation is outside the predeclared [0,4000) domain; do not clip/rebin")
    if not np.isin(soil, (0, 1)).all() or not np.all(soil.sum(axis=1) == 1):
        raise ValueError("Soil_Type is not a valid one-hot 40-category block")
    categories = np.column_stack((soil.argmax(axis=1), elevation // 100)).astype(np.int16)
    receipt = {"source_path": str(source), "source_sha256": source_hash,
               "archive_member": args.member if source.suffix == ".zip" else None,
               "source_text_sha256": text_hash, "source_text_bytes": len(raw),
               "shape": list(data.shape), "class_target_used": False,
               "elevation_min": int(elevation.min()), "elevation_max": int(elevation.max()),
               "bin_edges": "[0,100),[100,200),...,[3900,4000)",
               "category_sha256": array_sha(categories), "category_columns": ["soil", "elevation_band"]}
    path = out / "categories.npz"
    np.savez_compressed(path, categories=categories, metadata=np.array(json.dumps(receipt)))
    receipt["categories_file_sha256"] = sha_file(path)
    write_json(out / "prepared.json", receipt)
    return receipt


def load_categories(path):
    with np.load(path, allow_pickle=False) as archive:
        categories = np.array(archive["categories"], dtype=np.int16, copy=True)
        receipt = json.loads(str(archive["metadata"].item()))
    if categories.shape != (581012, 2) or np.any(categories < 0) or np.any(categories >= N):
        raise ValueError("Invalid prepared categorical data")
    if array_sha(categories) != receipt["category_sha256"]:
        raise ValueError("Prepared category content hash mismatch")
    return categories, receipt


def true_counts(categories):
    result = np.zeros((N, N), dtype=np.int64)
    np.add.at(result, (categories[:, 0], categories[:, 1]), 1)
    return result


def utility_expected_mse(records, enc):
    """Exact conditional sum of cell variances; sqrt(E[MSE]) is not E[RMSE]."""
    low, high = ((Fraction(1, 4), Fraction(1, 3)) if enc == "sue"
                 else (Fraction(1, 10), Fraction(1, 5)))
    low_count = 2 * (records // 4) + min(records % 4, 2)
    total = Fraction(0)
    for eta, count in ((low, low_count), (high, records - low_count)):
        p = 1 - eta if enc == "sue" else Fraction(1, 2)
        sigma0 = eta * (1 - eta) / (p - eta)**2
        sigma1 = p * (1 - p) / (p - eta)**2
        per_record = ((N - 1) * sigma0 + sigma1 + 1)**2 - 1
        total += count * per_record
    count_mse = total / (N * N)
    frequency_mse = count_mse / (records * records)
    return {"sum_cell_variance_fraction": str(total),
            "expected_count_mse_fraction": str(count_mse),
            "expected_frequency_mse_fraction": str(frequency_mse),
            "expected_count_mse": float(count_mse),
            "expected_frequency_mse": float(frequency_mse),
            "sqrt_expected_count_mse": math.sqrt(float(count_mse)),
            "sqrt_expected_frequency_mse": math.sqrt(float(frequency_mse)),
            "not_expected_rmse": True}


def acquire(args, out, deadline):
    plans, plan_sha, _ = load_plans(args.plans)
    cats, source = load_categories(args.categories)
    by_encoding = {}
    for p in plans:
        by_encoding.setdefault(p.encoding, p)
    if set(by_encoding) != {"sue", "oue"}:
        raise ValueError("Acquisition requires both SUE and matched-privacy OUE plans")
    ctx = Backend(args.backend, args.device)
    ids = parse_ids(args.stream_ids)
    if any(i not in range(20) for i in ids):
        raise ValueError("Predeclared acquisition stream IDs are 0..19")
    all_plan_fixed_ns = time.perf_counter_ns()
    manifest = {"metadata": metadata(), "backend": ctx.details, "plan_file_sha256": plan_sha,
                "source": source,
                "mode": "client_only_tape_acquisition" if args.client_only else "live_fresh_acquisition",
                "fresh_work_executions": 0 if args.client_only else len(ids),
                "all_plan_fixed_ns": all_plan_fixed_ns, "streams": []}
    with (out / "live_rounds.jsonl").open("x", encoding="utf-8") as rows:
        for sid in ids:
            deadline.check()
            # Predeclared disjoint source blocks; not selected by utility or timing.
            start = sid * 20000
            selected = cats[start:start + RECORDS]
            if len(selected) != RECORDS:
                raise ValueError("Insufficient rows in predeclared source block")
            live_plan = plans[sid % len(plans)]
            runtime = None if args.client_only else Runtime(ctx, live_plan, sid)
            tapes = {enc + suffix: np.empty((K, N, R), dtype=np.uint8)
                     for enc in by_encoding for suffix in ("_u", "_v")}
            client_common_ns = 0
            client_encoding_ns = dict.fromkeys(by_encoding, 0)
            round_records = []
            for j in range(K):
                deadline.check()
                generation_start = time.perf_counter_ns()
                draws = os_uniform(60, 2 * N * R).reshape(2, N, R)
                client_common_ns += time.perf_counter_ns() - generation_start
                pair = selected[R * j:R * (j + 1)]
                for enc, p in by_encoding.items():
                    tick = time.perf_counter_ns()
                    etas = (p.eta[j],) * R
                    tapes[enc + "_u"][j] = encode_from_uniform60(draws[0], pair[:, 0], enc, etas)
                    tapes[enc + "_v"][j] = encode_from_uniform60(draws[1], pair[:, 1], enc, etas)
                    client_encoding_ns[enc] += time.perf_counter_ns() - tick
                if args.client_only:
                    rec = {"kind": "client_round", "stream_id": sid, "round": j,
                           "all_plan_fixed_ns": all_plan_fixed_ns,
                           "input_generation_start_ns": generation_start,
                           "fresh_work_execution": False}
                else:
                    rec = runtime.process(j, tapes[live_plan.encoding + "_u"][j],
                                          tapes[live_plan.encoding + "_v"][j], deadline,
                                          input_generated_after_plan_ns=generation_start)
                    rec.update(performance_replay=False, security_input_scope="fresh OS client reports")
                round_records.append(rec)
                if not args.client_only and not rec["accepted"]:
                    for saved in round_records:
                        emit(rows, saved)
                    raise AssertionError("Honest live acquisition rejected; prescribed stream ended")
            tape_meta = {"stream_id": sid, "source_row_start": start,
                         "source_row_stop": start + RECORDS, "source_slice_sha256": array_sha(selected),
                         "plan_file_sha256": plan_sha,
                         "live_plan": None if args.client_only else live_plan.name,
                         "client_common_ns": client_common_ns, "client_encoding_ns": client_encoding_ns,
                         "input_rng": "OS exact uniform60 rejection after plan fixation",
                         "independent_os_input_realizations": 1,
                         "fresh_work_executions": 0 if args.client_only else 1,
                         "cross_encoding_coupling": "common random numbers for counterfactual benchmark comparison",
                         "array_sha256": {key: array_sha(value) for key, value in tapes.items()}}
            path = out / ("tape_%02d.npz" % sid)
            np.savez_compressed(path, **tapes, truth=true_counts(selected),
                                metadata=np.array(json.dumps(tape_meta)))
            tape_meta["file"] = path.name
            tape_meta["file_sha256"] = sha_file(path)
            tape_meta["live_decoded_num_sha256"] = None if args.client_only else array_sha(runtime.decoded_num)
            manifest["streams"].append(tape_meta)
            for rec in round_records:
                emit(rows, rec)
            write_json(out / "tapes_manifest.json", manifest)
    return {"completed_os_input_streams": len(manifest["streams"]),
            "fresh_work_executions": 0 if args.client_only else len(manifest["streams"]),
            "manifest": "tapes_manifest.json"}


def load_tapes(manifest_path):
    path = Path(manifest_path).resolve()
    obj = json.loads(path.read_text())
    records = sorted(obj["streams"], key=lambda x: x["stream_id"])
    if len({x["stream_id"] for x in records}) != len(records):
        raise ValueError("Duplicate report stream IDs")
    result = []
    for item in records:
        file = (path.parent / item["file"]).resolve()
        if file.parent != path.parent or sha_file(file) != item["file_sha256"]:
            raise ValueError("Tape path or file identity mismatch")
        with np.load(file, allow_pickle=False) as archive:
            arrays = {enc + suffix: np.array(archive[enc + suffix], copy=True)
                      for enc in ("sue", "oue") for suffix in ("_u", "_v")}
            truth = np.array(archive["truth"], copy=True)
            meta = json.loads(str(archive["metadata"].item()))
        for key, arr in arrays.items():
            if arr.shape != (K, N, R) or not np.isin(arr, (0, 1)).all():
                raise ValueError("Malformed report tape")
            if array_sha(arr) != meta["array_sha256"][key]:
                raise ValueError("Report tape array hash mismatch")
        result.append((item, meta, arrays, truth))
    return obj, result


def timing(args, out, deadline):
    plans, plan_sha, _ = load_plans(args.plans)
    plans = choose_plans(plans, args.plan_names)
    manifest, tapes = load_tapes(args.tapes)
    if manifest["plan_file_sha256"] != plan_sha:
        raise ValueError("Timing plans differ from the plans fixed before live input acquisition")
    allowed = set(parse_ids(args.stream_ids))
    tapes = [item for item in tapes if item[0]["stream_id"] in allowed]
    if not tapes:
        raise ValueError("No selected report tapes")
    ctx = Backend(args.backend, args.device)
    # One deterministic validation-only warmup per plan, outside primary timing.
    for p in plans:
        warm = Runtime(ctx, p, -1, test_mode=True)
        warm.process(0, np.zeros((N, R), dtype=np.uint8), np.zeros((N, R), dtype=np.uint8), deadline)
    ctx.sync()
    report_meta = {"metadata": metadata(), "backend": ctx.details,
                   "mode": "performance_replay", "plan_file_sha256": plan_sha,
                   "tapes_manifest_sha256": sha_file(args.tapes),
                   "source_os_input_stream_count": len(manifest["streams"]),
                   "source_fresh_work_executions": manifest.get("fresh_work_executions"),
                   "warmup": "one zero-factor round per plan; not a security episode",
                   "service_timer": "plan read/instance through final decoded output; client generation excluded",
                   "planning_cost_rule": "supplied one-time planner cost added once to complete 20-session totals",
                   "client_plus_service": "charged shared client cost plus observed replay service interval",
                   "plans": [{"name": p.name, "q": p.q, "encoding": p.encoding, "purpose": p.purpose,
                              "plan_sha256": p.digest, "planning_cost_ns": p.planning_cost_ns,
                              "planning_cost_source": p.planning_cost_source,
                              "total_checks": sum(p.t), "certificate": p.declared_certificate,
                              "wrong_increment_union_bound": str(sum((Fraction(1, p.q**t) for t in p.t), Fraction(0))),
                              "integrity_bound_is_distinct_from_work_certificate": True} for p in plans]}
    write_json(out / "timing_metadata.json", report_meta)
    completed = 0
    reference_by_encoding = {}
    with (out / "timing_rounds.jsonl").open("x", encoding="utf-8") as round_out, \
         (out / "timing_streams.jsonl").open("x", encoding="utf-8") as stream_out:
        for item, tape_meta, arrays, truth in tapes:
            sid = item["stream_id"]
            # A fixed cyclic Latin rotation; each plan occurs equally often in each position for 20/4.
            order = plans[sid % len(plans):] + plans[:sid % len(plans)]
            for position, expected_plan in enumerate(order):
                deadline.check()
                ctx.sync()
                total_start = time.perf_counter_ns()
                reread_start = total_start
                current, current_sha, _ = load_plans(args.plans)
                plan = next(p for p in current if p.name == expected_plan.name)
                if current_sha != plan_sha or plan.digest != expected_plan.digest:
                    raise ValueError("Frozen plan file changed during timing")
                load_ns = time.perf_counter_ns() - reread_start
                init_start = time.perf_counter_ns()
                runtime = Runtime(ctx, plan, sid)
                init_ns = time.perf_counter_ns() - init_start
                round_records = []
                for j in range(K):
                    rec = runtime.process(j, arrays[plan.encoding + "_u"][j],
                                          arrays[plan.encoding + "_v"][j], deadline)
                    rec.update(performance_replay=True, security_input_scope="reused reports; no new fresh-input episode")
                    round_records.append(rec)
                    if not rec["accepted"]:
                        for saved in round_records:
                            emit(round_out, saved)
                        raise AssertionError("Honest timing replay rejected; no replacement sample allowed")
                final_start = time.perf_counter_ns()
                final_num = np.array(runtime.decoded_num, dtype=np.int64, order="C", copy=True)
                final_decoded = final_num.astype(np.float64) / SCALE
                ctx.sync()
                final_ns = time.perf_counter_ns() - final_start
                total_ns = time.perf_counter_ns() - total_start
                # Validation/provenance and disk writes are outside the observed service interval.
                key = (sid, plan.encoding)
                result_sha = array_sha(final_num)
                if key in reference_by_encoding and reference_by_encoding[key] != result_sha:
                    raise AssertionError("Same reports yield different accepted decoded outputs across policies/fields")
                reference_by_encoding[key] = result_sha
                source_decode = np.zeros((N, N), dtype=np.int64)
                source_field = np.zeros((N, N), dtype=np.int64)
                for j in range(K):
                    uh, vh = arrays[plan.encoding + "_u"][j], arrays[plan.encoding + "_v"][j]
                    d = uh.astype(np.int64) @ vh.astype(np.int64).T
                    source_decode += decoded_delta(d, uh, vh, plan.encoding, plan.eta[j])
                    source_field = (source_field + d) % plan.q
                if not np.array_equal(source_decode, final_num) or not np.array_equal(source_field, runtime.host_c):
                    raise AssertionError("Independent integer report oracle mismatch")
                client_ns = tape_meta["client_common_ns"] + tape_meta["client_encoding_ns"][plan.encoding]
                receipt = {"kind": "stream", "stream_id": sid, "plan": plan.name,
                           "encoding": plan.encoding, "q": plan.q, "purpose": plan.purpose,
                           "backend": args.backend, "device": ctx.details, "order_position": position,
                           "performance_replay": True, "fresh_security_episodes_added": 0,
                           "report_tape_sha256": item["file_sha256"], "plan_sha256": plan.digest,
                           "service_total_ns": total_ns, "plan_read_instance_ns": load_ns,
                           "initialization_ns": init_ns, "final_decode_output_ns": final_ns,
                           "client_common_ns": tape_meta["client_common_ns"],
                           "client_encoding_ns": tape_meta["client_encoding_ns"][plan.encoding],
                           "client_plus_service_charged_ns": client_ns + total_ns,
                           "planning_cost_ns_once_per_20_session_batch": plan.planning_cost_ns,
                           "planning_cost_source": plan.planning_cost_source,
                           "accepted_rounds": runtime.accepted_rounds,
                           "committed_records": runtime.committed_records,
                           "checks_executed": sum(row["checks_attempted"] for row in round_records),
                           "checks_scheduled": sum(plan.t), "candidate_bytes_each": N * N * 4,
                           "decoded_num_sha256": result_sha, "decoded_scale": SCALE,
                           "unclipped_rmse": float(np.sqrt(np.mean((final_decoded - truth)**2))),
                           "negative_cells": int(np.count_nonzero(final_num < 0)),
                           "no_cross_field_security_equivalence_claim": True}
                emit(stream_out, receipt)
                for rec in round_records:
                    emit(round_out, rec)
                completed += 1
    return {"completed_method_streams": completed, "distinct_input_units": len(tapes),
            "complete_20_stream_grid": len(tapes) == 20 and completed == 20 * len(plans)}


def live_integrity(args, out, deadline):
    """Two genuinely fresh q65521 services: same source block, independent OS report coins."""
    all_plans, plan_sha, _ = load_plans(args.plans)
    plans = choose_plans([p for p in all_plans if p.q == 65521], args.plan_names)
    if not plans or any(p.purpose != "integrity" for p in plans):
        raise ValueError("live_integrity only executes the q65521 integrity plans")
    categories, source = load_categories(args.categories)
    ids = parse_ids(args.stream_ids)
    if any(sid not in range(20) for sid in ids):
        raise ValueError("Predeclared live stream IDs are 0..19")
    ctx = Backend(args.backend, args.device)
    for p in plans:
        warm = Runtime(ctx, p, -1, test_mode=True)
        warm.process(0, np.zeros((N, R), dtype=np.uint8), np.zeros((N, R), dtype=np.uint8), deadline)
    ctx.sync()
    meta = {"metadata": metadata(), "backend": ctx.details, "source": source,
            "mode": "live_integrity", "plan_file_sha256": plan_sha,
            "pairing": "source-block paired; independently OS-randomized for every plan/stream",
            "observed_live_total": "one continuous wall interval: plan read and initialization through final output",
            "active_service_time": "observed live interval minus disjoint measured client-generation spans; not a separate continuous wall interval",
            "freshness": "new independent report coins after each current plan fixation; challenges after each candidate freeze",
            "work_certificate_not_output_error": True,
            "plans": [{"name": p.name, "encoding": p.encoding, "q": p.q,
                       "plan_sha256": p.digest, "total_checks": sum(p.t),
                       "planning_cost_ns": p.planning_cost_ns,
                       "planning_cost_source": p.planning_cost_source,
                       "wrong_increment_union_bound": str(sum((Fraction(1, p.q**t) for t in p.t), Fraction(0)))}
                      for p in plans]}
    write_json(out / "integrity_live_metadata.json", meta)
    completed = 0
    with (out / "integrity_live_rounds.jsonl").open("x", encoding="utf-8") as round_out, \
         (out / "integrity_live_streams.jsonl").open("x", encoding="utf-8") as stream_out:
        for sid in ids:
            source_start = sid * 20000
            selected = categories[source_start:source_start + RECORDS]
            truth = true_counts(selected)
            order = plans[sid % len(plans):] + plans[:sid % len(plans)]
            for position, expected_plan in enumerate(order):
                deadline.check()
                ctx.sync()
                total_start = time.perf_counter_ns()
                reread, current_sha, _ = load_plans(args.plans)
                plan = next(p for p in reread if p.name == expected_plan.name)
                if current_sha != plan_sha or plan.digest != expected_plan.digest:
                    raise ValueError("Integrity plan changed before live execution")
                load_ns = time.perf_counter_ns() - total_start
                init_start = time.perf_counter_ns()
                runtime = Runtime(ctx, plan, sid)
                init_ns = time.perf_counter_ns() - init_start
                us = np.empty((K, N, R), dtype=np.uint8)
                vs = np.empty((K, N, R), dtype=np.uint8)
                client_ns = 0
                client_os_ns = 0
                round_records = []
                for j in range(K):
                    deadline.check()
                    generation_start = time.perf_counter_ns()
                    draws = os_uniform(60, 2 * N * R).reshape(2, N, R)
                    after_os = time.perf_counter_ns()
                    pair = selected[j * R:(j + 1) * R]
                    etas = (plan.eta[j],) * R
                    us[j] = encode_from_uniform60(draws[0], pair[:, 0], plan.encoding, etas)
                    vs[j] = encode_from_uniform60(draws[1], pair[:, 1], plan.encoding, etas)
                    generation_end = time.perf_counter_ns()
                    client_ns += generation_end - generation_start
                    client_os_ns += after_os - generation_start
                    rec = runtime.process(j, us[j], vs[j], deadline,
                                          input_generated_after_plan_ns=generation_start)
                    rec.update(performance_replay=False, fresh_work_execution=True,
                               security_input_scope="new OS reports after current plan fixation",
                               client_generation_ns=generation_end - generation_start,
                               client_generation_end_ns=generation_end)
                    round_records.append(rec)
                    if not rec["accepted"]:
                        for saved in round_records:
                            emit(round_out, saved)
                        raise AssertionError("Honest integrity live run rejected; sequence ended without replacement")
                final_start = time.perf_counter_ns()
                final_num = np.array(runtime.decoded_num, dtype=np.int64, order="C", copy=True)
                final_value = final_num.astype(np.float64) / SCALE
                ctx.sync()
                final_ns = time.perf_counter_ns() - final_start
                observed_ns = time.perf_counter_ns() - total_start
                active_ns = observed_ns - client_ns
                if active_ns <= 0:
                    raise AssertionError("Invalid client/service time decomposition")
                # Full integer oracle and preservation of raw inputs/outputs occur after timing.
                want_num = np.zeros((N, N), dtype=np.int64)
                want_field = np.zeros((N, N), dtype=np.int64)
                for j in range(K):
                    d = us[j].astype(np.int64) @ vs[j].astype(np.int64).T
                    want_num += decoded_delta(d, us[j], vs[j], plan.encoding, plan.eta[j])
                    want_field = (want_field + d) % plan.q
                if not np.array_equal(want_num, final_num) or not np.array_equal(want_field, runtime.host_c):
                    raise AssertionError("Fresh integrity run failed the outside-timer integer oracle")
                raw_path = out / ("live_%s_%02d.npz" % (plan.name, sid))
                np.savez_compressed(raw_path, u=us, v=vs, decoded_num=final_num,
                                    decoded_scale=np.array(SCALE, dtype=np.int64),
                                    decoded_estimate=final_value, truth=truth)
                row = {"kind": "live_integrity_stream", "stream_id": sid, "plan": plan.name,
                       "encoding": plan.encoding, "q": plan.q, "purpose": "integrity",
                       "backend": args.backend, "device": ctx.details, "order_position": position,
                       "performance_replay": False, "fresh_security_input_realizations": 1,
                       "pairing": "source paired, not report-coin paired",
                       "source_row_start": source_start, "source_row_stop": source_start + RECORDS,
                       "source_slice_sha256": array_sha(selected), "plan_sha256": plan.digest,
                       "u_sha256": array_sha(us), "v_sha256": array_sha(vs),
                       "decoded_num_sha256": array_sha(final_num), "raw_file": raw_path.name,
                       "raw_file_sha256": sha_file(raw_path),
                       "observed_live_total_ns": observed_ns, "client_generation_ns": client_ns,
                       "client_os_ns": client_os_ns, "active_service_ns": active_ns,
                       "active_service_is_discontinuous_measure": True,
                       "plan_read_instance_ns": load_ns, "initialization_ns": init_ns,
                       "final_decode_output_ns": final_ns,
                       "planning_cost_ns_once_per_20_session_batch": plan.planning_cost_ns,
                       "planning_cost_source": plan.planning_cost_source,
                       "accepted_rounds": runtime.accepted_rounds,
                       "committed_records": runtime.committed_records,
                       "checks_executed": sum(r["checks_attempted"] for r in round_records),
                       "checks_scheduled": sum(plan.t),
                       "wrong_increment_union_bound": str(sum((Fraction(1, plan.q**t) for t in plan.t), Fraction(0))),
                       "unclipped_rmse": float(np.sqrt(np.mean((final_value-truth)**2))),
                       "negative_cells": int(np.count_nonzero(final_num < 0)),
                       "noise_estimation_error_distinct_from_service_integrity": True}
                emit(stream_out, row)
                for rec in round_records:
                    emit(round_out, rec)
                completed += 1
    return {"completed_fresh_method_streams": completed, "source_blocks": len(ids),
            "complete_20_each": len(ids) == 20 and completed == 20 * len(plans)}


def utility(args, out, deadline):
    cats, source = load_categories(args.categories)
    plans, plan_sha, _ = load_plans(args.plans)
    enc_schedule = {}
    for plan in plans:
        enc_schedule.setdefault(plan.encoding, plan.eta)
    for enc, pair in (("sue", (Fraction(1, 4), Fraction(1, 3))),
                      ("oue", (Fraction(1, 10), Fraction(1, 5)))):
        if enc_schedule.get(enc) != pair * (K // 2):
            raise ValueError("Bulk utility's predeclared alternating schedule differs from the frozen plans")
    prefixes = sorted(set(int(x) for x in args.prefixes.split(",")))
    if prefixes != [16384, 65536, 262144, 581012]:
        raise ValueError("Use the predeclared four-prefix grid; do not tune it after seeing results")
    reps = parse_ids(args.replicate_ids)
    if any(i not in range(20) for i in reps):
        raise ValueError("Predeclared independent simulation replicate IDs are 0..19")
    truth = {stop: true_counts(cats[:stop]) for stop in prefixes}
    top20 = {stop: np.argsort(-value.reshape(-1), kind="stable")[:20]
             for stop, value in truth.items()}
    analytical = {enc: {stop: utility_expected_mse(stop, enc) for stop in prefixes}
                  for enc in ("sue", "oue")}
    np.savez_compressed(out / "prefix_truth.npz", **{"n_%d" % n: v for n, v in truth.items()})
    meta = {"metadata": metadata(), "source": source, "mode": "bulk_utility_simulation",
            "plan_file_sha256": plan_sha,
            "security_fresh_episodes": 0, "field_arithmetic_used": False,
            "input_rng": "NumPy PCG64 exact bounded integers 0..59; seeds fixed for simulation only",
            "analysis_seed": args.analysis_seed, "replicate_ids": reps, "prefixes": prefixes,
            "source_order": "unchanged, each prefix has its own exact target population",
            "cross_encoding_coupling": "common random numbers at equal per-attribute privacy",
            "privacy_scope": "each attribute epsilon ln9 or ln4; joint two-attribute loss sums",
            "utility_timer_excluded_from_service": True, "clipping": False,
            "decoder_scale": SCALE, "chunk_records": args.chunk_records,
            "analytic_reference": analytical,
            "analytic_comparison": "mean empirical MSE against exact E[MSE]; sqrt(E[MSE]) is not E[RMSE]",
            "top20_rule": "true-count descending; ties by fixed row-major cell index",
            "weighted_error_rule": "weights equal true cell count divided by prefix record count"}
    write_json(out / "utility_metadata.json", meta)
    finished = 0
    with (out / "utility_rows.jsonl").open("x", encoding="utf-8") as rows:
        for rep in reps:
            deadline.check()
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([args.analysis_seed, rep])))
            totals = {enc: np.zeros((N, N), dtype=np.int64) for enc in ("sue", "oue")}
            start = 0
            saved = {}
            rep_start = time.perf_counter_ns()
            for stop in prefixes:
                while start < stop:
                    deadline.check()
                    end = min(stop, start + args.chunk_records)
                    block = cats[start:end]
                    index = np.arange(start, end)
                    phase = (index // R) % 2
                    draws = rng.integers(0, 60, size=(2, N, end - start), dtype=np.uint8)
                    for enc in ("sue", "oue"):
                        low, high = ((Fraction(1, 4), Fraction(1, 3)) if enc == "sue"
                                     else (Fraction(1, 10), Fraction(1, 5)))
                        false_table = np.array([int(low * 60), int(high * 60)], dtype=np.uint8)
                        true_table = (np.array([int((1-low)*60), int((1-high)*60)], dtype=np.uint8)
                                      if enc == "sue" else np.array([30, 30], dtype=np.uint8))
                        f60, p60 = false_table[phase], true_table[phase]
                        u = encode_thresholds60(draws[0], block[:, 0], p60, f60)
                        v = encode_thresholds60(draws[1], block[:, 1], p60, f60)
                        for h, eta in enumerate((low, high)):
                            chosen = phase == h
                            count = int(np.count_nonzero(chosen))
                            if not count:
                                continue
                            us, vs = u[:, chosen], v[:, chosen]
                            # Binary dot sums <= chunk_records are exact in FP64; these are integer counts.
                            product = us.astype(np.float64) @ vs.astype(np.float64).T
                            if not np.array_equal(product, product.astype(np.int64)):
                                raise AssertionError("Bulk integer count representation failed")
                            a, b, c = decoder_coefficients(enc, eta)
                            mu, mv = us.sum(axis=1, dtype=np.int64), vs.sum(axis=1, dtype=np.int64)
                            totals[enc] += a * product.astype(np.int64) + b * (mu[:, None] + mv[None, :]) + c * count
                    start = end
                for enc in ("sue", "oue"):
                    estimate = totals[enc].astype(np.float64) / SCALE
                    error = estimate - truth[stop]
                    saved[enc + "_num_n_%d" % stop] = totals[enc].copy()
                    saved[enc + "_estimate_n_%d" % stop] = estimate.copy()
                    saved["truth_n_%d" % stop] = truth[stop]
                    saved["decoder_scale"] = np.array(SCALE, dtype=np.int64)
                    top_error = error.reshape(-1)[top20[stop]]
                    weights = truth[stop].astype(np.float64) / stop
                    mse = float(np.mean(error**2))
                    weighted_mse = float(np.sum(weights * error**2))
                    emit(rows, {"kind": "utility", "replicate_id": rep, "encoding": enc,
                                "records": stop, "source_prefix_sha256": array_sha(cats[:stop]),
                                "truth_sha256": array_sha(truth[stop]),
                                "estimate_num_sha256": array_sha(totals[enc]), "decoder_scale": SCALE,
                                "mse": mse, "frequency_mse": mse / (stop * stop),
                                "rmse": math.sqrt(mse), "frequency_rmse": math.sqrt(mse) / stop,
                                "mae": float(np.mean(np.abs(error))), "max_abs_error": float(np.max(np.abs(error))),
                                "true_top20_cell_indices": [int(x) for x in top20[stop]],
                                "true_top20_rmse": float(np.sqrt(np.mean(top_error**2))),
                                "true_top20_frequency_rmse": float(np.sqrt(np.mean(top_error**2))) / stop,
                                "true_top20_mae": float(np.mean(np.abs(top_error))),
                                "truth_weighted_rmse": math.sqrt(weighted_mse),
                                "truth_weighted_frequency_rmse": math.sqrt(weighted_mse) / stop,
                                "analytic_expected_mse": analytical[enc][stop],
                                "negative_cells": int(np.count_nonzero(totals[enc] < 0)),
                                "total_mass_error": float(estimate.sum() - stop),
                                "prefix_population": True, "clipped": False,
                                "bulk_simulation_elapsed_ns": time.perf_counter_ns() - rep_start})
            path = out / ("utility_rep_%02d.npz" % rep)
            np.savez_compressed(path, **saved)
            emit(rows, {"kind": "replicate_complete", "replicate_id": rep,
                        "estimate_file": path.name, "file_sha256": sha_file(path)})
            finished += 1
    return {"completed_replicates": finished, "requested_replicates": len(reps),
            "complete_20_replicates_in_this_worker": finished == 20}


def validate(args, out, deadline):
    plans, _, _ = load_plans(args.plans)
    ctx = Backend(args.backend, args.device)
    cases = []
    zero = np.zeros((N, R), dtype=np.uint8)
    base = plans[0]
    test_plan = dataclasses.replace(base, t=(2,) * K)
    vectors = [np.eye(N, dtype=np.uint32)[:, 0]] * 2
    rt = Runtime(ctx, test_plan, -1, test_mode=True)
    before = rt.fingerprint()
    row = rt.process(0, zero, zero, deadline, tamper_entry=(0, 0, 1), test_vectors=vectors)
    if row["accepted"] or before != rt.fingerprint() or row["reason"] != "freivalds_reject":
        raise AssertionError("Controlled rejecting vector/no-advance test failed")
    cases.append({"case": "controlled_reject_no_advance", "passed": True})
    rt = Runtime(ctx, test_plan, -2, test_mode=True)
    before = rt.fingerprint()
    row = rt.process(0, zero, zero, deadline, tamper_entry=(0, 0, R + 1), test_vectors=vectors)
    if row["accepted"] or before != rt.fingerprint() or row["reason"] != "increment_range":
        raise AssertionError("Out-of-range delta/no-advance test failed")
    cases.append({"case": "range_reject_no_advance", "passed": True})
    rt = Runtime(ctx, test_plan, -3, test_mode=True)
    row = rt.process(0, zero, zero, deadline, test_vectors=vectors, mutate_original=True)
    if not row["accepted"] or np.any(rt.host_c):
        raise AssertionError("Candidate snapshot isolation test failed")
    try:
        rt.host_c[0, 0] = 1
        raise AssertionError("Canonical received snapshot is mutable")
    except ValueError:
        pass
    cases.append({"case": "snapshot_tamper_isolation", "passed": True})
    try:
        Runtime(ctx, test_plan, -4).process(0, zero, zero, deadline, test_vectors=vectors)
        raise AssertionError("Production accepted a deterministic verifier hook")
    except ValueError:
        pass
    cases.append({"case": "test_hooks_forbidden_in_production", "passed": True})
    for enc, etas in (("sue", (Fraction(1, 4), Fraction(1, 3))),
                      ("oue", (Fraction(1, 10), Fraction(1, 5)))):
        uh = (np.arange(N * R).reshape(N, R) % 3 == 0).astype(np.uint8)
        vh = (np.arange(N * R).reshape(N, R) % 5 < 2).astype(np.uint8)
        d = uh.astype(np.int64) @ vh.astype(np.int64).T
        for eta in etas:
            p = 1 - eta if enc == "sue" else Fraction(1, 2)
            got = decoded_delta(d, uh, vh, enc, eta)
            for a in range(N):
                for b in range(N):
                    want = sum(((Fraction(int(uh[a, h])) - eta) *
                                (Fraction(int(vh[b, h])) - eta) / (p - eta)**2
                                for h in range(R)), Fraction(0))
                    if Fraction(int(got[a, b]), SCALE) != want:
                        raise AssertionError("Exact centered-product decoder oracle mismatch")
            deadline.check()
    cases.append({"case": "four_exact_decoders_vs_fraction_oracle", "passed": True})
    for q in (17, 65521):
        previous = np.full((N, N), q - 1, dtype=np.int64)
        increment = np.full((N, N), R, dtype=np.int64)
        candidate = (previous + increment) % q
        if not np.array_equal((candidate - previous) % q, increment):
            raise AssertionError("Cross-round modular wrap/lift test failed")
        a = np.full((3, N), q - 1, dtype=np.uint32)
        b = np.full((N, 4), q - 1, dtype=np.uint32)
        got = ctx.canonical(ctx.mm(ctx.array(a), ctx.array(b), q), q)
        if not np.all(got == (N * (q - 1)**2) % q):
            raise AssertionError("Exact field arithmetic oracle mismatch")
    cases.append({"case": "both_fields_exact_arithmetic_and_increment_lift", "passed": True})
    write_json(out / "validation.json", {"metadata": metadata(), "backend": ctx.details,
                                         "cases": cases, "empirical_security_rate_claim": False})
    return {"cases": len(cases), "all_passed": True}


def parse_ids(value):
    ids = [int(x) for x in value.split(",") if x.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("IDs must be a nonempty unique comma-separated list")
    return ids


def bootstrap_median(values, rng, count):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    bootstrap = np.median(values[indices], axis=1)
    return [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]


def summarize_live_integrity(args, out, deadline, path):
    meta = json.loads((path / "integrity_live_metadata.json").read_text())
    records = [json.loads(line) for line in (path / "integrity_live_streams.jsonl").read_text().splitlines() if line]
    grouped = {}
    for row in records:
        group = grouped.setdefault(row["plan"], {})
        if row["stream_id"] in group:
            raise ValueError("Duplicate live plan/stream measurement")
        group[row["stream_id"]] = row
    plan_meta = {p["name"]: p for p in meta["plans"]}
    methods = []
    for name, group in grouped.items():
        ids = sorted(group)
        if len(ids) != 20:
            raise ValueError("Live integrity summary requires twenty complete streams per plan")
        cost = plan_meta[name]["planning_cost_ns"]
        methods.append({"plan": name, "n_streams": len(ids),
                        "median_observed_live_ns": float(np.median([group[i]["observed_live_total_ns"] for i in ids])),
                        "median_active_service_ns": float(np.median([group[i]["active_service_ns"] for i in ids])),
                        "median_client_ns": float(np.median([group[i]["client_generation_ns"] for i in ids])),
                        "twenty_session_live_plus_once_planning_ns":
                        sum(group[i]["observed_live_total_ns"] for i in ids) + cost,
                        "twenty_session_active_plus_once_planning_ns":
                        sum(group[i]["active_service_ns"] for i in ids) + cost,
                        "planning_cost_ns_once": cost,
                        "wrong_increment_union_bound": plan_meta[name]["wrong_increment_union_bound"]})
    contrasts = []
    if "sue_integrity65521" in grouped and "oue_integrity65521" in grouped:
        sue, oue = grouped["sue_integrity65521"], grouped["oue_integrity65521"]
        ids = sorted(set(sue) & set(oue))
        if len(ids) != 20:
            raise ValueError("Missing source-paired live integrity blocks")
        points = []
        for i in ids:
            if sue[i]["source_slice_sha256"] != oue[i]["source_slice_sha256"]:
                raise ValueError("Live integrity encoding comparison is not source-paired")
            points.append({"stream_id": i,
                           "active_service_ratio_oue_over_sue": oue[i]["active_service_ns"] / sue[i]["active_service_ns"],
                           "observed_live_ratio_oue_over_sue": oue[i]["observed_live_total_ns"] / sue[i]["observed_live_total_ns"]})
        values = [x["active_service_ratio_oue_over_sue"] for x in points]
        rng = np.random.default_rng(args.analysis_seed)
        contrasts.append({"pairing": "twenty source blocks; independent report coins between encodings",
                          "same_coins": False, "same_q": 65521,
                          "points": points, "median_active_service_ratio": float(np.median(values)),
                          "nominal_95_percentile_bootstrap": bootstrap_median(values, rng, args.resamples),
                          "resamples": args.resamples, "analysis_seed": args.analysis_seed,
                          "not_a_cross_field_security_comparison": True})
    deadline.check()
    write_json(out / "summary.json", {"metadata": metadata(), "source_mode": "live_integrity",
                                       "source_rows_sha256": sha_file(path / "integrity_live_streams.jsonl"),
                                       "methods": methods, "contrasts": contrasts})
    return {"methods": len(methods), "contrasts": len(contrasts)}


def summarize(args, out, deadline):
    # This command is for the authorized remote analysis phase; never runs on import.
    path = Path(args.timing_dir)
    if (path / "integrity_live_metadata.json").is_file():
        return summarize_live_integrity(args, out, deadline, path)
    meta = json.loads((path / "timing_metadata.json").read_text())
    records = [json.loads(line) for line in (path / "timing_streams.jsonl").read_text().splitlines() if line]
    grouped = {}
    for row in records:
        group = grouped.setdefault(row["plan"], {})
        if row["stream_id"] in group:
            raise ValueError("Duplicate method/stream measurement")
        group[row["stream_id"]] = row
    plan_meta = {p["name"]: p for p in meta["plans"]}
    methods = []
    for name, group in grouped.items():
        ids = sorted(group)
        costs = [group[i]["service_total_ns"] for i in ids]
        planning = plan_meta[name]["planning_cost_ns"]
        methods.append({"plan": name, "n_streams": len(ids), "stream_ids": ids,
                        "service_median_ns": float(np.median(costs)),
                        "observed_service_sum_ns": sum(costs), "planning_cost_ns_once": planning,
                        "planning_inclusive_batch_ns": sum(costs) + planning,
                        "complete_twenty_sessions": len(ids) == 20})
    contrasts = []
    # Only same-encoding q17 policies are compared as resource-decision policies.
    opt_names = [name for name in grouped if name == "sue_opt17"]
    rr_names = [name for name in grouped if name == "sue_rr17"]
    if opt_names and rr_names:
        opt, rr = grouped[opt_names[0]], grouped[rr_names[0]]
        ids = sorted(set(opt) & set(rr))
        if len(ids) != 20:
            raise ValueError("Primary paired result requires all twenty complete input units")
        pc_opt, pc_rr = plan_meta[opt_names[0]]["planning_cost_ns"], plan_meta[rr_names[0]]["planning_cost_ns"]
        points = []
        for i in ids:
            if opt[i]["report_tape_sha256"] != rr[i]["report_tape_sha256"]:
                raise ValueError("Paired policies did not use the same tape")
            a, b = opt[i]["service_total_ns"], rr[i]["service_total_ns"]
            points.append({"stream_id": i, "service_ratio_rr_over_opt": b / a,
                           "amortized_planning_ratio": (b + pc_rr / 20) / (a + pc_opt / 20),
                           "service_difference_rr_minus_opt_ns": b - a,
                           "client_service_ratio": rr[i]["client_plus_service_charged_ns"] /
                                                   opt[i]["client_plus_service_charged_ns"]})
        rng = np.random.default_rng(args.analysis_seed)
        ratios = [x["amortized_planning_ratio"] for x in points]
        contrasts.append({"contrast": "fixed_prefix_RR_vs_exact_global_OPT_small_field",
                          "paired_points": points, "median_amortized_ratio": float(np.median(ratios)),
                          "nominal_95_percentile_bootstrap": bootstrap_median(ratios, rng, args.resamples),
                          "analysis_seed": args.analysis_seed, "resamples": args.resamples,
                          "twenty_session_planning_inclusive_ratio":
                          (sum(rr[i]["service_total_ns"] for i in ids) + pc_rr) /
                          (sum(opt[i]["service_total_ns"] for i in ids) + pc_opt),
                          "statistical_scope": "twenty paired randomizations on fixed source slices and observed schedules",
                          "cross_field_integrity_equivalence": False})
    result = {"metadata": metadata(), "source_metadata_sha256": sha_file(path / "timing_metadata.json"),
              "source_rows_sha256": sha_file(path / "timing_streams.jsonl"),
              "methods": methods, "contrasts": contrasts}
    deadline.check()
    write_json(out / "summary.json", result)
    return {"methods": len(methods), "contrasts": len(contrasts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("prepare", "validate", "acquire", "timing", "live_integrity", "utility", "summarize"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=1500)
    ap.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--source")
    ap.add_argument("--member", default="covtype.data.gz")
    ap.add_argument("--source-sha256", default=SOURCE_ZIP_SHA)
    ap.add_argument("--text-sha256", default=SOURCE_TEXT_SHA)
    ap.add_argument("--categories")
    ap.add_argument("--plans")
    ap.add_argument("--plan-names", help="Comma-separated existing plan names; timing/live_integrity only")
    ap.add_argument("--client-only", action="store_true", help="acquire tapes without executing a producer protocol")
    ap.add_argument("--tapes")
    ap.add_argument("--timing-dir")
    ap.add_argument("--stream-ids", default=",".join(str(x) for x in range(20)))
    ap.add_argument("--replicate-ids", default=",".join(str(x) for x in range(20)))
    ap.add_argument("--prefixes", default="16384,65536,262144,581012")
    ap.add_argument("--chunk-records", type=int, default=8192)
    ap.add_argument("--analysis-seed", type=int, default=20261110)
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    required = {"prepare": ("source",), "validate": ("plans",),
                "acquire": ("plans", "categories"), "timing": ("plans", "tapes"),
                "live_integrity": ("plans", "categories"),
                "utility": ("categories", "plans"), "summarize": ("timing_dir",)}
    for key in required[args.mode]:
        if not getattr(args, key):
            ap.error("--" + key.replace("_", "-") + " is required for this mode")
    if not 64 <= args.chunk_records <= 65536 or args.resamples < 1000:
        ap.error("Invalid bounded chunk/resampling size")
    deadline = Deadline(args.max_seconds)
    out = new_out(args.out)
    status = {"metadata": metadata(), "args": vars(args), "status": "running"}
    write_json(out / "status.json", status)
    try:
        result = {"prepare": prepare, "validate": validate, "acquire": acquire,
                  "timing": timing, "live_integrity": live_integrity,
                  "utility": utility, "summarize": summarize}[args.mode](args, out, deadline)
        status.update(status="completed", result=result, elapsed_seconds=time.monotonic() - deadline.start)
        status["files_sha256"] = {p.name: sha_file(p) for p in sorted(out.iterdir())
                                  if p.is_file() and p.name != "status.json"}
        write_json(out / "status.json", status)
    except Exception as exc:
        status.update(status="incomplete", error=type(exc).__name__ + ": " + str(exc),
                      elapsed_seconds=time.monotonic() - deadline.start)
        write_json(out / "status.json", status)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
