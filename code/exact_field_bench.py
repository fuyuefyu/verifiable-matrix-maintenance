#!/usr/bin/env python3
"""Exact q=65521 operator benchmark specification; primary and diagnostic modes."""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None
try:
    import resource
except ImportError:
    resource = None

Q = 65521
DEFAULT_SEED_IDS = "101,211,307,401,503"
METHODS = ("recompute_check", "incremental_check", "incremental_no_check")
PHASES = (
    "input_transfer", "common_A_update", "fixed_transform", "product",
    "candidate_update", "freeze", "candidate_receive", "randomness",
    "challenge_transfer", "verification", "state_advance",
)


def rng_for(seed, tag):
    ss = np.random.SeedSequence([int(seed), int(tag)])
    return np.random.Generator(np.random.PCG64(ss))


def rss_snapshot():
    current = hwm = None
    if psutil is not None:
        try:
            info = psutil.Process(os.getpid()).memory_info()
            current = int(info.rss)
            native_peak = getattr(info, "peak_wset", None)
            if native_peak is not None:
                hwm = int(native_peak)
        except (psutil.Error, OSError, AttributeError):
            pass
    if resource is not None:
        try:
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            hwm = int(value) * (1 if sys.platform == "darwin" else 1024)
        except (OSError, AttributeError):
            pass
    if current is None and sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as f:
                current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    return {"rss_current_bytes": current, "rss_process_hwm_bytes": hwm}


class Backend:
    def __init__(self, kind, device=0):
        self.kind, self.device = kind, device
        self.gpu = kind == "gpu"
        self.h2d = self.d2h = self.array_peak = 0
        self.xp, self.pool, self.name = np, None, platform.machine()
        if self.gpu:
            import cupy as cp
            self.xp = cp
            cp.cuda.Device(device).use()
            self.pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self._allocate)
            name = cp.cuda.runtime.getDeviceProperties(device)["name"]
            self.name = name.decode() if isinstance(name, bytes) else str(name)

    def _allocate(self, size):
        pointer = self.pool.malloc(size)
        self.array_peak = max(self.array_peak, int(self.pool.used_bytes()))
        return pointer

    def sync(self):
        if self.gpu:
            self.xp.cuda.runtime.deviceSynchronize()

    def array(self, host):
        """Copy canonical host residues to FP64 working storage."""
        host = np.ascontiguousarray(host)
        if self.gpu:
            self.h2d += int(host.nbytes)
            return self.xp.asarray(host).astype(self.xp.float64, copy=False)
        return np.array(host, dtype=np.float64, order="C", copy=True)

    def host(self, array):
        if self.gpu:
            self.d2h += int(array.nbytes)
            return self.xp.asnumpy(array)
        return np.asarray(array)

    def truth(self, scalar):
        if self.gpu:
            self.d2h += int(scalar.nbytes)
            return bool(scalar.item())
        return bool(scalar)

    def reset_peak(self):
        self.array_peak = int(self.pool.used_bytes()) if self.gpu else 0

    def memory(self):
        result = rss_snapshot()
        if self.gpu:
            free, total = self.xp.cuda.runtime.memGetInfo()
            result.update({
                "gpu_array_live_bytes": int(self.pool.used_bytes()),
                "gpu_array_peak_bytes": int(self.array_peak),
                "gpu_pool_reserved_bytes": int(self.pool.total_bytes()),
                "gpu_device_used_bytes": int(total - free),
            })
        return result


def modmatmul(ctx, left, right):
    """Exact dot products of canonical FP64 residues under the checked bound."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("modmatmul: incompatible two-dimensional operands")
    d = int(left.shape[1])
    if d <= 0 or d * (Q - 1) ** 2 >= 2 ** 53:
        raise ValueError("FP64 exact-integer accumulation bound exceeded")
    if left.dtype != ctx.xp.float64 or right.dtype != ctx.xp.float64:
        raise TypeError("modmatmul requires FP64 working arrays")
    product = ctx.xp.matmul(left, right)
    return ctx.xp.remainder(product.astype(ctx.xp.int64), Q).astype(ctx.xp.float64)


def modadd(ctx, left, right):
    """Canonical operands and signed differences of them are supported."""
    value = left.astype(ctx.xp.int64) + right.astype(ctx.xp.int64)
    return ctx.xp.remainder(value, Q).astype(ctx.xp.float64)


def canonical(ctx, array):
    """Independent read-only, row-major, little-endian uint32 host snapshot."""
    encoded = array.astype(ctx.xp.uint32)
    valid = ctx.xp.all((array >= 0) & (array < Q) & (array == encoded))
    if not ctx.truth(valid):
        raise ValueError("Noncanonical or nonintegral field output")
    host = np.array(ctx.host(encoded), dtype="<u4", order="C", copy=True)
    host.flags.writeable = False
    return host


def matrix_digest(host, context=None):
    h = hashlib.sha256()
    h.update(f"field-matrix-v1:{Q}:{host.shape}:<u4:C;".encode("ascii"))
    if context is not None:
        encoded = json.dumps(context, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")
        h.update(b"context:")
        h.update(encoded)
        h.update(b"\0")
    h.update(memoryview(host).cast("B"))
    return h.hexdigest()


def freeze(ctx, candidate, context=None):
    host = canonical(ctx, candidate)
    return host, matrix_digest(host, context)


def fresh_vectors(n, t):
    """Unbiased OS randomness: accept exactly the uint16 values below Q."""
    out = np.empty(n * t, dtype="<u4")
    used = 0
    while used < out.size:
        draw = np.frombuffer(os.urandom(2 * (out.size - used + 8)), dtype="<u2")
        accepted = draw[draw < Q]
        take = min(accepted.size, out.size - used)
        out[used:used + take] = accepted[:take]
        used += take
    return out.reshape(n, t)


def check(ctx, candidate, previous, u, vt, b, z):
    """The caller must freeze candidate before generating independent z."""
    lhs = modmatmul(ctx, modadd(ctx, candidate, -previous), z)
    bz = modmatmul(ctx, b, z)
    vbz = modmatmul(ctx, vt, bz)
    rhs = modmatmul(ctx, u, vbz)
    return ctx.truth(ctx.xp.all(lhs == rhs))


class Clock:
    def __init__(self, ctx, profile_sync=False):
        self.ctx, self.profile_sync = ctx, bool(profile_sync)
        self.ns = dict.fromkeys(PHASES, 0)

    def call(self, name, function):
        start = time.perf_counter_ns()
        result = function()
        if self.profile_sync:
            self.ctx.sync()
        self.ns[name] += time.perf_counter_ns() - start
        return result


def python_selftest(ctx, inner):
    rng = rng_for(123456789, 0x5151)
    ah = rng.integers(0, Q, size=(3, inner), dtype=np.uint32)
    bh = rng.integers(0, Q, size=(inner, 4), dtype=np.uint32)
    ah[0, :], bh[:, 0] = Q - 1, Q - 1
    got = canonical(ctx, modmatmul(ctx, ctx.array(ah), ctx.array(bh)))
    for i in range(3):
        for j in range(4):
            want = sum(int(ah[i, k]) * int(bh[k, j]) for k in range(inner)) % Q
            if int(got[i, j]) != want:
                raise AssertionError("Independent Python-integer GEMM selftest failed")


def prepare(ctx, n, seed):
    start = time.perf_counter_ns()
    rng = rng_for(seed, 0xBACE)
    ah = rng.integers(0, Q, size=(n, n), dtype=np.uint32)
    lower = np.tril(rng.integers(0, Q, size=(n, n), dtype=np.uint32))
    upper = np.triu(rng.integers(0, Q, size=(n, n), dtype=np.uint32))
    np.fill_diagonal(lower, 1)
    np.fill_diagonal(upper, 1)
    a0 = ctx.array(ah)
    b = modmatmul(ctx, ctx.array(lower), ctx.array(upper))
    c0 = modmatmul(ctx, a0, b)
    ctx.sync()
    setup_ns = time.perf_counter_ns() - start
    diagnostic_start = time.perf_counter_ns()
    chosen = rng.choice(n * n, size=min(4, n * n), replace=False)
    coords = [(int(x) // n, int(x) % n) for x in chosen]
    cols = {j: np.array(ctx.host(b[:, j]), dtype=np.uint32) for _, j in coords}
    spots = {(i, j): int(ah[i, j]) for i, j in coords}
    for i, j in coords:
        want = sum(int(ah[i, k]) * int(cols[j][k]) for k in range(n)) % Q
        if int(ctx.host(c0[i, j])) != want:
            raise AssertionError("Initial-product bigint check failed")
    return {
        "a0": a0, "b": b, "c0": c0, "coords": coords, "cols": cols,
        "spots": spots, "setup_ns": setup_ns,
        "setup_validation_ns": time.perf_counter_ns() - diagnostic_start,
    }


def execute_round(ctx, state, b, uh, vth, t, method, profile_sync=False,
                  round_context=None):
    if method not in METHODS:
        raise ValueError("Unknown method")
    a, previous = state
    ctx.sync()
    memory_before = ctx.memory()
    ctx.reset_peak()
    tx0, rx0 = ctx.h2d, ctx.d2h
    clock = Clock(ctx, profile_sync)
    start = time.perf_counter_ns()
    u, vt = clock.call("input_transfer", lambda: (ctx.array(uh), ctx.array(vth)))
    an = clock.call("common_A_update", lambda: modadd(ctx, a, modmatmul(ctx, u, vt)))
    transformed = None
    if method != "recompute_check":
        transformed = clock.call("fixed_transform", lambda: modmatmul(ctx, vt, b))
    product = clock.call(
        "product", lambda: modmatmul(ctx, an, b) if method == "recompute_check"
        else modmatmul(ctx, u, transformed),
    )
    candidate = clock.call(
        "candidate_update", lambda: product if method == "recompute_check"
        else modadd(ctx, previous, product),
    )
    frozen, digest = clock.call("freeze", lambda: freeze(ctx, candidate, round_context))
    received = clock.call("candidate_receive", lambda: ctx.array(frozen))
    zh = z = None
    accepted = True
    if method != "incremental_no_check":
        zh = clock.call("randomness", lambda: fresh_vectors(a.shape[0], t))
        z = clock.call("challenge_transfer", lambda: ctx.array(zh))
        accepted = clock.call(
            "verification", lambda: check(ctx, received, previous, u, vt, b, z)
        )
    if accepted:
        def advance():
            state[:] = [an, received]
        clock.call("state_advance", advance)
    del u, vt, transformed, product, candidate, z
    ctx.sync()
    total_ns = time.perf_counter_ns() - start
    residual = total_ns - sum(clock.ns.values())
    if residual < 0:
        raise AssertionError("Negative timing residual")
    clock.ns["orchestration"] = residual
    row = {
        "method": method, "accepted": accepted, "total_ns": total_ns,
        "total_ms": total_ns / 1e6,
        "timing_mode": "profile_sync" if profile_sync else "primary_e2e",
        "h2d_payload_bytes": ctx.h2d - tx0, "d2h_payload_bytes": ctx.d2h - rx0,
        "memory_before": memory_before, "memory_after": ctx.memory(),
        "candidate_sha256": digest, "candidate_context": round_context,
        "challenge_sha256": None if zh is None else matrix_digest(zh),
        "canonical_state_bytes": int(3 * a.shape[0] ** 2 * 4),
        "resident_state_array_bytes": int(3 * a.shape[0] ** 2 * 8),
        "reset_snapshot_array_bytes": int(2 * a.shape[0] ** 2 * 8),
    }
    if ctx.gpu and not profile_sync:
        row["phase_time_semantics"] = "host_observed_calls_and_necessary_waits"
        prefix = "host_observed_phase"
    else:
        row["phase_time_semantics"] = (
            "synchronized_stage_wall_time" if ctx.gpu else "cpu_stage_wall_time"
        )
        prefix = "component_phase"
    row[prefix + "_ns"] = clock.ns
    row[prefix + "_ms"] = {key: value / 1e6 for key, value in clock.ns.items()}
    return row, frozen, zh


def validate_round(ctx, state, base, frozen, uh, vth, expected_spots):
    a, c = state
    oracle = modmatmul(ctx, a, base["b"])
    full_match = ctx.truth(ctx.xp.all(oracle == c))
    spot_match = True
    for i, j in base["coords"]:
        delta = sum(int(uh[i, k]) * int(vth[k, j]) for k in range(uh.shape[1]))
        expected_spots[(i, j)] = (expected_spots[(i, j)] + delta) % Q
        arow = np.array(ctx.host(a[i, :]), dtype=np.uint32)
        spot_match &= int(arow[j]) == expected_spots[(i, j)]
        want = sum(int(arow[k]) * int(base["cols"][j][k]) for k in range(a.shape[0])) % Q
        spot_match &= int(frozen[i, j]) == want
    ctx.sync()
    return bool(full_match), bool(spot_match)


def warmup(ctx, args, seed, deadline):
    deadline()
    base = prepare(ctx, args.n, seed)
    deadline()
    rng = rng_for(seed, 0xFACA)
    uh = rng.integers(0, Q, size=(args.n, args.r), dtype=np.uint32)
    vth = rng.integers(0, Q, size=(args.r, args.n), dtype=np.uint32)
    context = {"warmup": True, "seed": seed, "repeat": 0, "round": 0,
               "n": args.n, "r": args.r, "t": args.t}
    for method in METHODS:
        deadline()
        state = [base["a0"].copy(), base["c0"].copy()]
        row, frozen, _ = execute_round(
            ctx, state, base["b"], uh, vth, args.t, method,
            profile_sync=args.profile_sync, round_context=context,
        )
        good = validate_round(ctx, state, base, frozen, uh, vth, dict(base["spots"]))
        if not row["accepted"] or not all(good):
            raise AssertionError("Independent-state warmup failed")
        deadline()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--r", type=int, required=True)
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-ids", default=DEFAULT_SEED_IDS)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Start index in --seed-ids, not an arithmetic seed offset")
    parser.add_argument("--profile-sync", action="store_true",
                        help="Diagnostic stage barriers; exclude from primary latency plots")
    parser.add_argument("--max-seconds", type=float, default=6600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not (1 <= args.r <= args.n <= 4096):
        parser.error("Require 1 <= r <= n <= 4096")
    if min(args.t, args.rounds, args.seeds, args.repeats) < 1 or args.seed_offset < 0:
        parser.error("Counts must be positive and seed-offset nonnegative")
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds must be finite and positive")
    try:
        seed_ids = [int(token.strip()) for token in args.seed_ids.split(",")]
    except ValueError:
        parser.error("--seed-ids must be comma-separated nonnegative integers")
    if not seed_ids or any(seed < 0 for seed in seed_ids):
        parser.error("--seed-ids must contain nonnegative integers")
    if len(set(seed_ids)) != len(seed_ids):
        parser.error("--seed-ids must not contain duplicate seeds")
    seed_end = args.seed_offset + args.seeds
    if seed_end > len(seed_ids):
        parser.error("--seed-offset plus --seeds exceeds the supplied seed list")
    selected_seed_ids = seed_ids[args.seed_offset:seed_end]
    warm_seed = max(seed_ids + [987654321]) + 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.monotonic()
    completed = 0
    with open(args.out, "x", encoding="utf-8", buffering=1) as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

        def deadline():
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError("Cooperative wall budget reached; includes validation")

        try:
            ctx = Backend(args.backend, args.device)
            emit({
                "kind": "metadata", "version": "exact-field-bench-v2", "args": vars(args),
                "q": Q, "hostname": platform.node(), "device_name": ctx.name,
                "numpy_version": np.__version__, "backend_version": ctx.xp.__version__,
                "selected_seed_ids": selected_seed_ids, "warmup_seed": warm_seed,
                "timing_mode": "profile_sync" if args.profile_sync else "primary_e2e",
                "extra_stage_barriers": bool(ctx.gpu and args.profile_sync),
                "component_times_available": bool(not ctx.gpu or args.profile_sync),
                "input_rng": "PCG64 deterministic benchmark fixtures",
                "verifier_rng": "OS uint16 rejection sampling after candidate freeze",
                "oracle_scope": "same-backend full AB plus Python-bigint spot checks",
                "memory_scope": "CuPy live-array peak; RSS HWM is process-lifetime",
                "transport_scope": "actual local host/device transfers; no network emulation",
                "deadline_scope": "cooperative operation boundaries; external hard timeout required",
                "total_time_scope": "input transfer through state advance, including necessary sync",
                "thread_env": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            })
            deadline()
            python_selftest(ctx, args.n)
            deadline()
            warm_start = time.perf_counter_ns()
            warmup(ctx, args, warm_seed, deadline)
            gc.collect()
            emit({"kind": "warmup", "wall_ms": (time.perf_counter_ns() - warm_start) / 1e6})
            deadline()
            for si, seed in enumerate(selected_seed_ids):
                deadline()
                base = prepare(ctx, args.n, seed)
                emit({"kind": "setup", "seed": seed, "common_setup_ns": base["setup_ns"],
                      "setup_validation_ns": base["setup_validation_ns"]})
                deadline()
                for repeat in range(args.repeats):
                    seed_list_index = args.seed_offset + si
                    rotation = (seed_list_index * args.repeats + repeat) % len(METHODS)
                    order = METHODS[rotation:] + METHODS[:rotation]
                    for position, method in enumerate(order):
                        reset_start = time.perf_counter_ns()
                        state = [base["a0"].copy(), base["c0"].copy()]
                        ctx.sync()
                        reset_ns = time.perf_counter_ns() - reset_start
                        emit({"kind": "trial_reset", "seed": seed, "repeat": repeat,
                              "method": method, "order_position": position, "reset_ns": reset_ns})
                        expected = dict(base["spots"])
                        rng = rng_for(seed, 0xFACA)
                        for round_id in range(args.rounds):
                            deadline()
                            gen_start = time.perf_counter_ns()
                            uh = rng.integers(0, Q, (args.n, args.r), dtype=np.uint32)
                            vth = rng.integers(0, Q, (args.r, args.n), dtype=np.uint32)
                            gen_ns = time.perf_counter_ns() - gen_start
                            context = {"seed": seed, "repeat": repeat, "round": round_id,
                                       "n": args.n, "r": args.r, "t": args.t}
                            old_state = tuple(state)
                            row, frozen, zh = execute_round(
                                ctx, state, base["b"], uh, vth, args.t, method,
                                profile_sync=args.profile_sync, round_context=context,
                            )
                            row.update({"kind": "round", "seed": seed, "repeat": repeat,
                                        "round": round_id, "order_position": position,
                                        "n": args.n, "r": args.r, "t": args.t,
                                        "input_generation_ns_outside_timer": gen_ns})
                            if not row["accepted"]:
                                state[:] = old_state
                                row["failed_check_vectors"] = None if zh is None else zh.tolist()
                                row["trial_valid"] = False
                                emit(row)
                                raise AssertionError("Verifier rejected an intended exact candidate")
                            validation_start = time.perf_counter_ns()
                            try:
                                full_ok, bigint_ok = validate_round(
                                    ctx, state, base, frozen, uh, vth, expected
                                )
                            except Exception:
                                state[:] = old_state
                                raise
                            row.update({
                                "validation_ns_outside_timer": time.perf_counter_ns() - validation_start,
                                "full_AB_match": full_ok, "python_bigint_spots_match": bigint_ok,
                                "trial_valid": bool(full_ok and bigint_ok),
                            })
                            if not (full_ok and bigint_ok):
                                state[:] = old_state
                                row["diagnostic_rollback"] = True
                                emit(row)
                                raise AssertionError("Full-state or bigint validation failed")
                            emit(row)
                            completed += 1
                            del frozen, zh, old_state
                            deadline()
                        del state
                del base
                gc.collect()
                deadline()
            deadline()
            emit({"kind": "completed", "round_records": completed,
                  "wall_seconds": time.monotonic() - started})
        except Exception as error:
            emit({"kind": "failed", "error_type": type(error).__name__,
                  "error": str(error), "completed_round_records": completed,
                  "wall_seconds": time.monotonic() - started})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
