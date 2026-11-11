#!/usr/bin/env python3
"""Linux-only, bounded executor for fixed argv plans; no scientific analysis."""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

UTC = dt.timezone.utc
GRACE = 10.0


def stamp():
    return dt.datetime.now(UTC).isoformat()


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= GRACE:
        raise ValueError(label + " must be finite and exceed the 10-second termination allowance")
    return float(value)


def load_plan(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan.jobs must be a nonempty list")
    ids = set()
    for job in jobs:
        for key in ("id", "group"):
            if not isinstance(job.get(key), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job[key]):
                raise ValueError("Invalid job " + key)
        if job["id"] in ids:
            raise ValueError("Duplicate job id: " + job["id"])
        ids.add(job["id"])
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x or "\x00" in x for x in argv):
            raise ValueError("argv must be a nonempty string list")
        env = job.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in env.items()):
            raise ValueError("env must contain explicit non-secret string values")
        if any(re.search(r"TOKEN|PASSWORD|SECRET|API_KEY|AUTHORIZATION", k, re.I) for k in env):
            raise ValueError("Secret environment values must not be included in a recorded plan")
        if job.get("cwd") is not None and not Path(job["cwd"]).is_dir():
            raise ValueError("Missing job cwd: " + str(job["cwd"]))
        positive(job.get("timeout_seconds"), "timeout_seconds")
    return plan, hashlib.sha256(raw).hexdigest()


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--max-seconds", type=float, required=True)
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        ap.error("This runner supports Linux process groups only; Windows needs a separate launcher")
    positive(args.max_seconds, "max-seconds")
    absolute = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
    if absolute.tzinfo is None:
        ap.error("deadline-utc must include a timezone")
    absolute = absolute.astimezone(UTC)
    plan, sha = load_plan(args.plan)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    start = time.monotonic()
    allowance = min(args.max_seconds, (absolute - dt.datetime.now(UTC)).total_seconds())
    hard_end = start + max(0, allowance)
    soft_end = hard_end - GRACE
    state = {"schema":"icde04-grid-execution-v1", "owner":owner, "controller_pid":os.getpid(),
             "hostname":os.uname().nodename, "plan_sha256":sha, "started_utc":stamp(),
             "deadline_utc":absolute.isoformat(), "max_seconds":args.max_seconds,
             "inherited_environment_recorded":False, "status":"running", "reason":None,
             "jobs":[{"spec":j,"status":"not-started"} for j in plan["jobs"]]}
    pending = collections.OrderedDict()
    for index, job in enumerate(plan["jobs"]):
        pending.setdefault(job["group"], collections.deque()).append(index)
    active = {}
    stop = {"reason":None, "exit":0}

    def save():
        temp = out / "manifest.writing.json"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(out / "manifest.json")

    def request_stop(reason, code):
        if stop["reason"] is None:
            stop.update(reason=reason, exit=code)

    def on_signal(number, frame):
        request_stop("signal-" + str(number), 128 + number)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, on_signal)

    def close_handles(entry):
        for handle in entry["handles"]:
            handle.close()

    def terminate_owned(reason):
        if not active:
            return
        now = time.monotonic()
        for entry in active.values():
            row = state["jobs"][entry["index"]]
            row.update(termination_requested_utc=stamp(), termination_reason=reason)
            try:
                os.killpg(entry["pgid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        kill_at = min(now + GRACE, hard_end)
        while time.monotonic() < kill_at and any(group_alive(e["pgid"]) for e in active.values()):
            for entry in active.values():
                entry["process"].poll()
            time.sleep(0.1)
        for group, entry in list(active.items()):
            row = state["jobs"][entry["index"]]
            if group_alive(entry["pgid"]):
                try:
                    os.killpg(entry["pgid"], signal.SIGKILL)
                    row["sigkill_sent_utc"] = stamp()
                except ProcessLookupError:
                    pass
            try:
                code = entry["process"].wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = None
            close_handles(entry)
            alive = group_alive(entry["pgid"])
            row.update(status="timeout" if "timeout" in reason or "deadline" in reason else "terminated",
                       exit_code=code, ended_utc=stamp(),
                       wall_seconds=time.monotonic()-entry["started"],
                       owned_process_group_absent=not alive)
            if alive:
                row["cleanup_error"] = "Owned process group still visible; do not claim closure"
            del active[group]

    save()
    try:
        while active or any(pending.values()):
            now = time.monotonic()
            if now >= soft_end or dt.datetime.now(UTC) >= absolute - dt.timedelta(seconds=GRACE):
                request_stop("batch-deadline", 124)
            if stop["reason"] is not None:
                break
            for group, entry in list(active.items()):
                row = state["jobs"][entry["index"]]; job = row["spec"]
                code = entry["process"].poll()
                if code is not None:
                    if group_alive(entry["pgid"]):
                        row["leader_exit_code"] = code
                        request_stop("orphaned-owned-descendants:" + job["id"], 1)
                        break
                    close_handles(entry)
                    row.update(status="completed" if code == 0 else "failed", exit_code=code,
                               ended_utc=stamp(), wall_seconds=time.monotonic()-entry["started"],
                               owned_process_group_absent=True)
                    del active[group]
                    save()
                    if code != 0:
                        request_stop("job-failed:" + job["id"], 1)
                        break
                elif time.monotonic()-entry["started"] >= job["timeout_seconds"]-GRACE:
                    request_stop("job-timeout:" + job["id"], 124)
                    break
            if stop["reason"] is not None:
                break
            for group, queue in pending.items():
                if group in active or not queue:
                    continue
                if time.monotonic() >= soft_end or stop["reason"] is not None:
                    break
                index = queue.popleft(); job = plan["jobs"][index]; row = state["jobs"][index]
                handles = ((out / (job["id"] + ".stdout")).open("xb"),
                           (out / (job["id"] + ".stderr")).open("xb"))
                env = os.environ.copy(); env.update(job.get("env", {}))
                env["ICDE04_GRID_OWNER"] = owner
                launched = time.monotonic()
                try:
                    proc = subprocess.Popen(job["argv"], cwd=job.get("cwd"), env=env,
                                            stdout=handles[0], stderr=handles[1], start_new_session=True)
                except Exception as error:
                    for handle in handles: handle.close()
                    row.update(status="failed", error=str(error), ended_utc=stamp())
                    request_stop("launch-failed:" + job["id"], 1)
                    break
                active[group] = {"index":index,"process":proc,"pgid":proc.pid,
                                 "handles":handles,"started":launched}
                row.update(status="running", pid=proc.pid, owned_pgid=proc.pid,
                           started_utc=stamp(), owner=owner)
                save()
            time.sleep(0.1)
    except Exception as error:
        request_stop("controller-error:" + str(error), 1)
    finally:
        terminate_owned(stop["reason"] or "controller-finalization")
        for row in state["jobs"]:
            if row["status"] == "not-started":
                row["not_started_reason"] = stop["reason"] or "not-dispatched"
        complete = all(row["status"] == "completed" for row in state["jobs"])
        state.update(status="completed" if complete else "incomplete", reason=stop["reason"],
                     ended_utc=stamp(), wall_seconds=time.monotonic()-start,
                     all_owned_groups_closed=all(row.get("owned_process_group_absent",True) for row in state["jobs"]))
        save()
    print(json.dumps({"status":state["status"],"jobs":len(state["jobs"]),
                      "completed":sum(x["status"]=="completed" for x in state["jobs"]),
                      "all_owned_groups_closed":state["all_owned_groups_closed"]}))
    return 0 if complete and state["all_owned_groups_closed"] else (stop["exit"] or 1)


if __name__ == "__main__":
    raise SystemExit(main())
