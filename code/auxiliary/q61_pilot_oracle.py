#!/usr/bin/env python3
"""Independent, deliberately slow Python-bigint check of a small q61 trace.

Run only on the deployment host after the C++ pilot. This file has only
been syntax-checked by its author; it does not modify benchmark records.
"""
import hashlib
import json
import sys
from pathlib import Path

Q = (1 << 61) - 1
METHODS = ("recompute_check", "incremental_check", "incremental_no_check")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def matrix(obj):
    n, m = obj["rows"], obj["cols"]
    require(isinstance(n, int) and isinstance(m, int) and 0 <= n <= 64 and 0 <= m <= 64,
            "invalid trace matrix dimensions")
    a = [int(x) for x in obj["data"]]
    require(len(a) == n * m and all(0 <= x < Q for x in a), "invalid field entries")
    return [a[i * m:(i + 1) * m] for i in range(n)]


def trans(a):
    return [list(x) for x in zip(*a)] if a else []


def mm(a, b):
    require(a and b and len(a[0]) == len(b), "bigint matrix shape mismatch")
    # An unbounded Python-integer sum, reduced only after the complete dot.
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) % Q
             for j in range(len(b[0]))] for i in range(len(a))]


def combine(a, b, sign=1):
    require(len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b)),
            "bigint addition shape mismatch")
    return [[(x + sign * y) % Q for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def det(a):
    a = [row[:] for row in a]
    n, result = len(a), 1
    require(n and all(len(row) == n for row in a), "B is not square")
    for k in range(n):
        pivot = next((i for i in range(k, n) if a[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            result = -result % Q
        value = a[k][k]
        result = result * value % Q
        inverse = pow(value, Q - 2, Q)
        for i in range(k + 1, n):
            factor = a[i][k] * inverse % Q
            for j in range(k, n):
                a[i][j] = (a[i][j] - factor * a[k][j]) % Q
    return result


def check(path):
    raw = path.read_bytes()
    trace = json.loads(raw)
    require(trace.get("version") == "q61-pilot-trace-v1", "unexpected trace version")
    require(int(trace["q"]) == Q and trace.get("status") == "completed", "trace is not a completed q61 pilot")
    cases, updates, records, check_count = {}, {}, 0, None
    for event in trace["events"]:
        key = (event["seed"], event["repeat"])
        if event["kind"] == "setup":
            require(key not in cases, "duplicate setup")
            a0, b, c0 = (matrix(event[k]) for k in ("A0", "B", "C0"))
            require(1 <= len(a0) <= 16 and any(x for row in a0 for x in row), "invalid or trivial A0")
            require(det(b) == 1, "B does not have the determinant of unit-L times unit-U")
            require(mm(a0, b) == c0, "initial C0 != A0 B")
            cases[key] = {"B": b, "states": {m: ([row[:] for row in a0], [row[:] for row in c0])
                                             for m in METHODS}, "next": {m: 0 for m in METHODS}}
            continue
        require(event["kind"] == "round" and key in cases, "missing setup or unknown event")
        case, method, round_id = cases[key], event["method"], event["round"]
        require(method in METHODS and round_id == case["next"][method], "out-of-order or duplicate round")
        require(event["accepted"] is True and event["advanced"] is True, "pilot did not accept and advance")
        u, v = matrix(event["U"]), matrix(event["V"])
        update_key = key + (round_id,)
        if update_key in updates:
            require(updates[update_key] == (u, v), "methods did not receive matched update factors")
        else:
            updates[update_key] = (u, v)
        old_a, old_c = case["states"][method]
        expected_a = combine(old_a, mm(u, trans(v)))
        expected_c = mm(expected_a, case["B"])
        delivered = matrix(event["C_delivered"])
        require(matrix(event["A_attempted"]) == expected_a, "common A update mismatch")
        require(delivered == expected_c, "delivered candidate != full bigint A B")
        zs = matrix(event["check_vectors"])
        if method == "incremental_no_check":
            require(event["check_performed"] is False and event["check_pass"] is None and not zs,
                    "nocheck method reported a check")
        else:
            require(event["check_performed"] is True and event["check_pass"] is True and zs,
                    "checked method has missing or failed checks")
            if check_count is None:
                check_count = len(zs)
            require(len(zs) == check_count, "inconsistent number of check vectors")
            z = trans(zs)
            lhs = mm(combine(delivered, old_c, -1), z)
            rhs = mm(u, mm(trans(v), mm(case["B"], z)))
            require(lhs == rhs, "independent verification equation mismatch")
        require(matrix(event["post_A"]) == expected_a and matrix(event["post_C"]) == expected_c,
                "state advance mismatch")
        case["states"][method] = (expected_a, expected_c)
        case["next"][method] += 1
        records += 1
    require(cases and records, "empty pilot")
    for case in cases.values():
        counts = set(case["next"].values())
        require(len(counts) == 1 and 1 <= next(iter(counts)) <= 2, "incomplete matched pilot")
    return {"kind": "python_bigint_verification", "version": "q61-pilot-bigint-v1",
            "status": "completed", "q": str(Q), "trace_sha256": hashlib.sha256(raw).hexdigest(),
            "setup_cases": len(cases), "round_records": records,
            "python_bigint_full_match": True, "B_determinants_one": True,
            "matching_inputs": True, "check_equations_match": True, "state_advance_match": True}


if __name__ == "__main__":
    try:
        require(len(sys.argv) == 2, "usage: q61_pilot_oracle.py TRACE.json")
        print(json.dumps(check(Path(sys.argv[1])), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"kind": "python_bigint_verification", "version": "q61-pilot-bigint-v1",
                          "status": "failed", "reason": str(exc)}, sort_keys=True))
        sys.exit(1)
