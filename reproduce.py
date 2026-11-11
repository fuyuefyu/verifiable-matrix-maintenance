#!/usr/bin/env python3
"""Portable, local-only orchestration of the frozen ICDE artifact (Linux).

Install this file and reproduce_config.json at the artifact root. Keep the exact
sources under code/, archived host plans under plans/{grid,stream}_plans/, the
source archive at data/covertype.zip, and historical stream data beside
data/historical/stream_manifest.json. No dependencies or data are downloaded.

Examples (replace the explicit UTC timestamps with one future allocation):
  python reproduce.py --profile planner --deadline-utc 2027-01-01T10:00:00Z \
      --max-seconds 900 --out reruns/planner-01 --dry-run
  python reproduce.py --profile connected-setup --deadline-utc UTC_DEADLINE \
      --max-seconds 1800 --out reruns/connected-setup-01
  python reproduce.py --profile connected-cpu --deadline-utc SAME_DEADLINE \
      --max-seconds 1800 --connected-inputs reruns/connected-setup-01 \
      --out reruns/connected-cpu-01

Copy the whole completed setup directory manually before running GPU or utility
profiles on another host. Every dependent profile inherits that setup's batch
start and deadline: it does not receive a fresh 30-minute allowance. Run only
one connected profile at a time on each physical host; utility and timing must
not overlap there. For main/extension host profiles, explicitly share
--batch-start-utc and --deadline-utc across the entire <=12-hour allocation.

--dry-run verifies identities and writes a new derived command manifest without
importing any scientific kernel. Actual execution uses the frozen run_grid.py
controller for owned process groups and hard deadlines. This entrypoint does
not resume, retry, delete partial results, fetch credentials, or invoke SSH.

The planner's historical HARD_STOP remains unchanged in its source. A worker
loads those exact verified bytes under a non-main module name, records the one
runtime-global override, then calls the original main(). All other science is
executed through the original command interfaces. Kernel receipts keep their
original source hash; the wrapper receipt additionally binds the runtime policy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time

UTC = dt.timezone.utc
MAX_ALLOCATION = 43200
MAX_CONNECTED = 1800
GRACE = 10
SCHEMA = "icde04-portable-reproduction-v1"
RECEIPT = "REPRODUCE_RECEIPT.json"
SETUP_FILES = ("planner/plans.json", "planner/PLANNER_RECEIPT.json",
               "planner/exact_certificates.json", "prepared/categories.npz",
               "prepared/prepared.json", "tapes/tapes_manifest.json")


def utc(value):
    if not isinstance(value, str):
        raise ValueError("UTC timestamps must be explicit ISO-8601 strings")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise ValueError("Use an explicit UTC timestamp ending in Z or +00:00")
    return result.astimezone(UTC)


def stamp(value=None):
    return (value or dt.datetime.now(UTC)).isoformat().replace("+00:00", "Z")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def positive(value, label, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be numeric")
    if not math.isfinite(value) or not GRACE < value <= upper:
        raise ValueError(f"{label} must be in ({GRACE},{upper}]")
    return float(value)


def relative_path(root, value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Artifact paths must be nonempty relative strings")
    part = Path(value)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Artifact paths cannot be absolute or contain '..': " + value)
    result = (root / part).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Artifact path escapes its root through a symlink: " + value)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    tmp.replace(path)


def executable(value):
    found = shutil.which(value)
    if found is None:
        raise ValueError("Executable unavailable; install/configure it separately: " + value)
    # Keep the venv entrypoint spelling: resolving its symlink to /usr/bin/python
    # would silently discard the selected environment's site-packages.
    return os.path.abspath(found)


def verify_file(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("A complete lowercase SHA-256 identity is required")
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"Frozen identity mismatch: {path}; expected {expected}, got {actual}")
    return actual


class Sources:
    def __init__(self, root, config):
        self.root, self.config, self.used = root, config, {}

    def need(self, name):
        spec = self.config["sources"][name]
        path = relative_path(self.root, spec["path"])
        if name not in self.used:
            self.used[name] = {"path": str(path), "artifact_path": spec["path"],
                               "sha256": verify_file(path, spec["sha256"])}
        return path


def load_exact_module(path, expected, name):
    """Execute verified source bytes, without using a possibly cached .pyc."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Source changed before loading: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ValueError("Cannot load frozen module: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def worker(argv):
    """Internal run_grid job entry: verify again immediately before execution."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--runtime-sha256", required=True)
    ap.add_argument("--job", required=True)
    args = ap.parse_args(argv)
    if not os.environ.get("ICDE04_GRID_OWNER"):
        raise ValueError("Internal workers must be launched by the bounded controller")
    verify_file(args.runtime, args.runtime_sha256)
    runtime = read_json(args.runtime)
    verify_file(__file__, runtime["wrapper_sha256"])
    end = utc(runtime["effective_deadline_utc"])
    if dt.datetime.now(UTC) >= end:
        raise TimeoutError("The shared reproduction deadline has expired")
    jobs = {job["id"]: job for job in runtime["jobs"]}
    job = jobs[args.job]
    checked = {}
    for name in job["sources"]:
        entry = runtime["source_catalog"][name]
        checked[name] = verify_file(entry["path"], entry["sha256"])
    for entry in job.get("data_bindings", []):
        verify_file(entry["path"], entry["sha256"])
    launch = {"job_id": job["id"], "started_utc": stamp(), "pid": os.getpid(),
              "owned_group": os.getpgrp(), "runtime_sha256": args.runtime_sha256,
              "source_hashes": checked, "argv": job["argv"],
              "effective_deadline_utc": stamp(end), "science_modified": False}
    launch_path = Path(runtime["output_root"]) / "launches" / (job["id"] + ".json")
    if job["kind"] == "planner":
        entry = runtime["source_catalog"]["planner"]
        module = load_exact_module(entry["path"], entry["sha256"], "_icde04_frozen_planner")
        before = getattr(module, "HARD_STOP", None)
        if before != runtime["frozen_planner_hard_stop"]:
            raise ValueError("Unexpected frozen planner HARD_STOP; refuse an unreviewed override")
        after = end.timestamp()
        # Only this runtime global is changed. The frozen file is never written.
        module.HARD_STOP = after
        launch["runtime_override"] = {"name": "HARD_STOP", "frozen_value": before,
                                      "runtime_value": after,
                                      "method": "verified-module-global injection before original main"}
        write_json(launch_path, launch)
        sys.argv = [entry["path"], *job["argv"][2:]]
        result = module.main()
        return 0 if result is None else int(result)
    if job["kind"] == "q61":
        launch["compiled_binary_sha256"] = sha_file(job["argv"][0])
    write_json(launch_path, launch)
    # exec preserves the controller-owned process group and its hard timeout.
    os.execvpe(job["argv"][0], job["argv"], os.environ.copy())
    raise AssertionError("exec unexpectedly returned")


def one_flag(argv, flag, value):
    if argv.count(flag) != 1:
        raise ValueError("Archived argv must contain exactly one " + flag)
    at = argv.index(flag)
    if at + 1 >= len(argv):
        raise ValueError("Archived flag has no value: " + flag)
    argv[at + 1] = value


def data_binding(root, config, name):
    spec = config["data"][name]
    path = relative_path(root, spec["path"])
    return {"name": name, "artifact_path": spec["path"], "path": str(path),
            "sha256": verify_file(path, spec["sha256"])}


def build_archived(root, out, profile, config, sources, python, cxx, env):
    declaration = config["archived_plans"][profile]
    path = relative_path(root, declaration["path"])
    old_sha = verify_file(path, declaration["sha256"])
    original = read_json(path)
    jobs, relocation, q61_group = [], [], None
    for frozen in original["jobs"]:
        old = frozen["argv"]
        family = frozen["family"]
        bindings = []
        if family == "q61":
            if Path(old[0]).name != "q61_bench":
                raise ValueError("Unknown archived q61 command")
            sources.need("q61_cpp")
            q61_group = q61_group or frozen["group"]
            if frozen["group"] != q61_group:
                raise ValueError("q61 build ordering requires one declared local group")
            command = [str(out / "bin/q61_bench"), *old[1:]]
            used, kind = ["q61_cpp"], "q61"
        else:
            name = {"synthetic": "exact_field", "covertype": "covertype"}.get(family)
            if name is None or len(old) < 2:
                raise ValueError("Unknown archived scientific family: " + str(family))
            script = sources.need(name)
            if Path(old[1]).name != script.name:
                raise ValueError("Archived script name differs from the bound source")
            used = [name]
            if name == "covertype":
                sources.need("exact_field")
                used.append("exact_field")
                bindings.append(data_binding(root, config, "stream_manifest"))
            command, kind = [python, str(script), *old[2:]], "python"
            if bindings:
                one_flag(command, "--manifest", bindings[0]["path"])
        original_output = old[old.index("--out") + 1]
        one_flag(command, "--out", str(out / "raw" / Path(original_output).name))
        job_env = dict(env, CUPY_CACHE_DIR=str(out / "cache" / frozen["group"]))
        jobs.append({"id": frozen["id"], "group": frozen["group"],
                     "kind": kind, "argv": command, "sources": used,
                     "data_bindings": bindings, "env": job_env,
                     "timeout_seconds": frozen["timeout_seconds"],
                     "expected_round_records": frozen.get("expected_round_records"),
                     "family": family})
        relocation.append({"job": frozen["id"], "frozen_argv": old,
                           "derived_argv": command, "frozen_cwd": frozen.get("cwd"),
                           "frozen_env": frozen.get("env", {}), "derived_env": job_env})
    if q61_group is not None:
        compiler = executable(cxx)
        command = [compiler, "-O3", "-std=c++17", "-Wall", "-Wextra", "-Wshadow",
                   str(sources.need("q61_cpp")), "-o", str(out / "bin/q61_bench")]
        jobs.insert(0, {"id": "build_q61", "group": q61_group, "kind": "compile",
                        "argv": command, "sources": ["q61_cpp"], "env": env,
                        "timeout_seconds": 140, "compiler_sha256": sha_file(compiler)})
    return jobs, {"path": str(path), "sha256": old_sha,
                  "original_host_alias": original.get("host_alias"),
                  "historical_deadline_utc": original.get("deadline_utc"),
                  "relocations": relocation}


def setup_inputs(root, relative, config_sha, sources):
    path = relative_path(root, relative)
    receipt_path = path / RECEIPT
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "completed" or receipt.get("profile") != "connected-setup"
            or receipt.get("config_sha256") != config_sha):
        raise ValueError("Connected inputs require a completed setup under this exact configuration")
    for name in ("planner", "connected", "runner"):
        sources.need(name)
        if receipt.get("source_catalog", {}).get(name, {}).get("sha256") != sources.used[name]["sha256"]:
            raise ValueError("Connected setup source identity differs: " + name)
    expected = receipt.get("setup_output_bindings", {})
    if set(expected) != set(SETUP_FILES):
        raise ValueError("The completed setup receipt lacks its exact output bindings")
    bindings = [{"path": str(path / name), "sha256": verify_file(path / name, expected[name])}
                for name in SETUP_FILES]
    bindings.append({"path": str(receipt_path), "sha256": sha_file(receipt_path)})
    # Detailed plan/tape/array binding is additionally enforced by the frozen harness.
    return path, receipt, bindings


def build_profiles(root, out, declaration, config, sources, python, env, variables):
    jobs = []
    for job in declaration["jobs"]:
        name = job["source"]
        path = sources.need(name)
        bindings = [data_binding(root, config, x) for x in job.get("data", [])]
        substitutions = dict(variables)
        substitutions.update({item["name"]: item["path"] for item in bindings})
        command = [python, str(path), *[arg.format_map(substitutions) for arg in job["args"]]]
        jobs.append({"id": job["id"], "group": "local_serial",
                     "kind": "planner" if name == "planner" else "python",
                     "argv": command, "sources": [name], "data_bindings": bindings,
                     "env": dict(env, CUPY_CACHE_DIR=str(out / "cache/local_serial")),
                     "timeout_seconds": job["timeout_seconds"]})
    return jobs


def main(argv=None):
    started = dt.datetime.now(UTC)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--config", default="reproduce_config.json", help="Relative to artifact root")
    ap.add_argument("--profile", required=True, help="Named profile or archived host plan in the config")
    ap.add_argument("--deadline-utc", required=True, help="Explicit UTC; no later than allocation start +12 hours")
    ap.add_argument("--max-seconds", required=True, type=float, help="Local wall cap, additionally bounded by the shared deadline")
    ap.add_argument("--batch-start-utc", help="Shared allocation start; required for all main/stream host profiles")
    ap.add_argument("--out", required=True, help="New directory relative to artifact root; never reused")
    ap.add_argument("--connected-inputs", help="Completed connected setup directory, relative to artifact root")
    ap.add_argument("--raw-input", help="One unmixed raw-result tree, relative to artifact root; summarize-grid only")
    ap.add_argument("--python", default=sys.executable, help="Existing Python environment; no installation")
    ap.add_argument("--cxx", default="g++", help="Existing compiler for archived q61 cells")
    ap.add_argument("--device", type=int, default=0, help="Local connected GPU device; archived grid IDs remain unchanged")
    ap.add_argument("--cuda-path", help="Optional local toolkit path; otherwise preserve an existing CUDA_PATH")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not sys.platform.startswith("linux"):
        ap.error("The frozen process-group controller requires Linux")
    if args.device < 0:
        ap.error("device must be nonnegative")
    root = Path(args.artifact_root).resolve(strict=True)
    config_path = relative_path(root, args.config)
    config = read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise ValueError("Unsupported reproduction configuration schema")
    if not 0 < config["maximum_allocation_seconds"] <= MAX_ALLOCATION:
        raise ValueError("Configuration cannot extend the 12-hour allocation")
    if not 0 < config["connected_batch_seconds"] <= MAX_CONNECTED:
        raise ValueError("Configuration cannot extend the 30-minute connected batch")
    requested = utc(args.deadline_utc)
    if not started < requested <= started + dt.timedelta(seconds=MAX_ALLOCATION):
        raise ValueError("deadline-utc must be in the future and at most 12 hours after startup")
    frozen_end = utc(config["frozen_deadline_utc"])
    if frozen_end.timestamp() != config["frozen_planner_hard_stop"]:
        raise ValueError("Frozen planner timestamp and recorded historical UTC disagree")
    if not args.dry_run and started <= frozen_end and requested > frozen_end:
        raise ValueError("This future-run entrypoint cannot extend the still-active historical deadline")
    config_sha = sha_file(config_path)
    sources = Sources(root, config)
    runner_path = sources.need("runner")
    archived = args.profile in config["archived_plans"]
    declaration = (config["archived_plans"] if archived else config["profiles"]).get(args.profile)
    if declaration is None:
        raise ValueError("Unknown profile: " + args.profile)
    local_cap = positive(args.max_seconds, "max-seconds",
                         min(MAX_ALLOCATION, declaration["wall_cap_seconds"]))
    out = relative_path(root, args.out)
    if out.exists():
        raise ValueError("Output root already exists; retain it and choose a new directory")
    if archived and not args.batch_start_utc:
        raise ValueError("Main/extension host profiles require a shared --batch-start-utc")
    batch_start = utc(args.batch_start_utc) if args.batch_start_utc else started
    inputs, inherited, input_bindings = None, None, []
    if declaration.get("requires_setup"):
        if not args.connected_inputs:
            raise ValueError("This profile requires --connected-inputs from a completed setup")
        inputs, inherited, input_bindings = setup_inputs(root, args.connected_inputs, config_sha, sources)
        inherited_start = utc(inherited["batch_started_utc"])
        if args.batch_start_utc and batch_start != inherited_start:
            raise ValueError("Dependent connected profiles cannot reset the setup's batch start")
        batch_start = inherited_start
    elif args.connected_inputs:
        raise ValueError("--connected-inputs applies only to dependent connected profiles")
    if batch_start > started:
        raise ValueError("batch-start-utc must not lie in the future")
    if requested > batch_start + dt.timedelta(seconds=config["maximum_allocation_seconds"]):
        raise ValueError("Requested deadline exceeds the shared allocation's 12-hour bound")
    batch_cap = config["connected_batch_seconds"] if declaration.get("connected") else config["maximum_allocation_seconds"]
    batch_deadline = min(requested, batch_start + dt.timedelta(seconds=batch_cap))
    if inherited is not None:
        batch_deadline = min(batch_deadline, utc(inherited["batch_deadline_utc"]))
    effective = min(batch_deadline, started + dt.timedelta(seconds=local_cap))
    if (effective - dt.datetime.now(UTC)).total_seconds() <= GRACE:
        raise ValueError("The shared budget is exhausted; no new local allowance is created")
    python = executable(args.python)
    env = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(runner_path.parent)}
    cuda = args.cuda_path if args.cuda_path is not None else os.environ.get("CUDA_PATH")
    if cuda:
        env["CUDA_PATH"] = str(Path(cuda).resolve())
    variables = {"out": str(out), "inputs": str(inputs or out), "device": str(args.device)}
    if declaration.get("requires_raw_input"):
        if not args.raw_input:
            raise ValueError("summarize-grid requires --raw-input for one unmixed collection")
        raw = relative_path(root, args.raw_input)
        if not raw.is_dir():
            raise ValueError("Raw-result tree is unavailable")
        variables["raw_input"] = str(raw)
    elif args.raw_input:
        raise ValueError("--raw-input applies only to summarize-grid")
    archived_receipt = None
    if archived:
        jobs, archived_receipt = build_archived(root, out, args.profile, config, sources, python, args.cxx, env)
    else:
        jobs = build_profiles(root, out, declaration, config, sources, python, env, variables)
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Derived jobs must have nonempty unique identities")
    wrapper = Path(__file__).resolve()
    wrapper_sha = sha_file(wrapper)
    runtime = {"schema": SCHEMA, "profile": args.profile, "artifact_root": str(root),
               "output_root": str(out), "config_sha256": config_sha,
               "wrapper_sha256": wrapper_sha, "source_catalog": sources.used,
               "started_utc": stamp(started), "batch_started_utc": stamp(batch_start),
               "requested_deadline_utc": stamp(requested), "batch_deadline_utc": stamp(batch_deadline),
               "effective_deadline_utc": stamp(effective), "max_seconds": local_cap,
               "frozen_planner_hard_stop": config["frozen_planner_hard_stop"],
               "frozen_deadline_utc": config["frozen_deadline_utc"],
               "connected_input_bindings": input_bindings,
               "original_argv": sys.argv if argv is None else [str(wrapper), *argv],
               "python": {"path": python, "sha256": sha_file(python)},
               "controller_python": sys.version, "hostname": socket.gethostname(),
               "archived_plan": archived_receipt, "jobs": jobs,
               "inherited_environment_recorded": False,
               "deadline_override_policy": "Only planner.HARD_STOP is injected; exact frozen source bytes and all scientific algorithms remain unchanged"}
    # All preflight validation precedes claiming the new owned output root.
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("work", "raw", "bin", "launches", "cache"):
        (out / directory).mkdir()
    for job in jobs:
        (out / "cache" / job["group"]).mkdir(exist_ok=True)
    (out / "reproduce_config.used.json").write_bytes(config_path.read_bytes())
    runtime_path = out / "runtime.json"
    write_json(runtime_path, runtime)
    runtime_sha = sha_file(runtime_path)
    derived_jobs = []
    for job in jobs:
        derived_jobs.append({"id": job["id"], "group": job["group"],
                             "argv": [python, str(wrapper), "_worker", "--runtime", str(runtime_path),
                                      "--runtime-sha256", runtime_sha, "--job", job["id"]],
                             "cwd": str(out / "work"), "env": job["env"],
                             "timeout_seconds": job["timeout_seconds"],
                             "expected_round_records": job.get("expected_round_records")})
    plan_path = out / "derived_plan.json"
    write_json(plan_path, {"schema": "icde04-fixed-argv-portable-v1", "deadline_utc": stamp(effective), "jobs": derived_jobs})
    receipt = {key: value for key, value in runtime.items() if key not in ("jobs", "archived_plan")}
    receipt.update(status="dry-run" if args.dry_run else "running",
                   runtime_sha256=runtime_sha, derived_plan_sha256=sha_file(plan_path),
                   controller_started=False, expected_job_ids=ids,
                   completion_scope="Controller completion is separate from scientific coverage, correctness and acceptance")
    write_json(out / RECEIPT, receipt)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "jobs": len(jobs), "receipt": str(out / RECEIPT),
                          "effective_deadline_utc": stamp(effective)}))
        return 0
    result = 1
    try:
        remaining = (effective - dt.datetime.now(UTC)).total_seconds()
        if remaining <= GRACE:
            raise TimeoutError("Preflight consumed the remaining budget; no kernel was launched")
        sys.dont_write_bytecode = True
        runner = load_exact_module(runner_path, sources.used["runner"]["sha256"], "_icde04_frozen_runner")
        sys.argv = [str(runner_path), "--plan", str(plan_path), "--out", str(out / "controller"),
                    "--deadline-utc", stamp(effective), "--max-seconds", str(remaining)]
        receipt.update(controller_started=True, controller_argv=sys.argv)
        write_json(out / RECEIPT, receipt)
        result = int(runner.main())
        controller = read_json(out / "controller/manifest.json")
        by_id = {row["spec"]["id"]: row for row in controller["jobs"]}
        complete = (result == 0 and controller.get("status") == "completed"
                    and controller.get("all_owned_groups_closed") is True
                    and set(by_id) == set(ids)
                    and all(row.get("status") == "completed" for row in by_id.values()))
        receipt.update(status="completed" if complete else "incomplete",
                       controller_manifest_sha256=sha_file(out / "controller/manifest.json"),
                       all_owned_groups_closed=controller.get("all_owned_groups_closed"),
                       job_statuses={name: row.get("status") for name, row in by_id.items()})
        if complete and declaration.get("setup"):
            receipt["setup_output_bindings"] = {name: sha_file(out / name) for name in SETUP_FILES}
        if not complete:
            result = result or 1
    except Exception as error:
        receipt.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
        result = 1
    finally:
        receipt.update(ended_utc=stamp(), exit_code=result)
        write_json(out / RECEIPT, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(out / RECEIPT), "exit_code": result}))
    return result


if __name__ == "__main__":
    try:
        code = worker(sys.argv[2:]) if len(sys.argv) > 1 and sys.argv[1] == "_worker" else main()
    except (ValueError, OSError, KeyError, TypeError, TimeoutError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        code = 2
    raise SystemExit(code)
