"""Job worker: SQS -> agent -> DynamoDB (+ Step Functions human approval when the agent escalates or a flood
threshold is crossed). Runs on Graviton (stock OpenCV 5 or COOL).

A job is a JSON message:
    {"job_id": "...", "site": "lamorge"}                              # a dataset site already on the worker
    {"job_id": "...", "video_s3": "s3://bucket/uploads/x.mp4",        # an uploaded, orthorectified clip
     "fps": 30, "gsd": 0.01, "water": [[x, y], ...], "targets": [[x, y], ...], "flood_q": 5.0}
Optional: "planner": "rules" | "claude".

    python -m gaugeless.worker --queue-url URL --table T --state-machine ARN      # on AWS
    python -m gaugeless.worker --local '{"job_id": "t1", "site": "lamorge"}'      # no AWS, prints the record
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import traceback
from decimal import Decimal

import cv2
import numpy as np

from .agent import tools as T
from .agent.llm import _clean
from .agent.run import make_session
from .cases import Case


def _video_session(job, workdir):
    import boto3
    bucket, key = job["video_s3"][5:].split("/", 1)
    path = os.path.join(workdir, os.path.basename(key))
    boto3.client("s3").download_file(bucket, key, path)
    fdir = os.path.join(workdir, "frames")
    os.makedirs(fdir)
    cap = cv2.VideoCapture(path)
    i = 0
    while i < 600:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(fdir, f"{i:05d}.png"), f)
        i += 1
    cap.release()
    fps = float(job.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30)
    water = tuple(tuple(p) for p in job["water"]) if job.get("water") else None
    case = Case("upload", fdir, fps, float(job["gsd"]), None, "", "upload", water=water)
    s = T.Session(case=case, flood_q=job.get("flood_q"))
    if job.get("targets"):
        s.targets = np.asarray(job["targets"], float)
    return s


def process(job: dict, verbose=False) -> dict:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as wd:
        s = make_session(job["site"], job.get("flood_q")) if job.get("site") else _video_session(job, wd)
        call = lambda tool, **a: T.TOOLS[tool](s, **a)
        planner = job.get("planner") or os.environ.get("GAUGELESS_PLANNER", "rules")
        job["planner"] = planner
        if planner == "claude":
            from .agent import llm
            llm.plan(s, call, verbose=verbose)
        else:
            from .agent import rules
            rules.plan(s, call)
    outcome = "published" if s.published else "refused" if s.refused else "escalated" if s.escalated else "none"
    n_frames = max((t["result"].get("frames", 0) for t in s.trace if t["tool"] == "load_clip"), default=0)
    dt = time.time() - t0
    rec = {"job_id": job["job_id"], "site": job.get("site") or "upload", "outcome": outcome,
           "planner": job.get("planner", "rules"), "seconds": round(dt, 2),
           "ms_per_frame": round(1000 * dt / max(n_frames, 1), 2), "frames": n_frames,
           "published": s.published, "refused": s.refused, "escalated": s.escalated,
           "discharge": {k: v for k, v in (s.discharge or {}).items() if k != "sections"} or None,
           "trace": s.trace, "host": platform.machine(), "opencv": cv2.__version__,
           "llm_usage": getattr(s, "llm_usage", None), "summary": getattr(s, "llm_summary", None)}
    return _clean(rec)


def _ddb(o):
    if isinstance(o, float):
        return Decimal(str(o))
    if isinstance(o, dict):
        return {k: _ddb(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_ddb(v) for v in o]
    return o


def serve(queue_url, table, state_machine=None, region=None):
    import boto3
    sqs = boto3.client("sqs", region_name=region)
    ddb = boto3.resource("dynamodb", region_name=region).Table(table)
    sfn = boto3.client("stepfunctions", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    while True:
        r = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=20, VisibilityTimeout=900)
        for m in r.get("Messages", []):
            job = json.loads(m["Body"])
            if job.get("Event") == "s3:TestEvent":      # sent once when the bucket notification is set up
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m["ReceiptHandle"]); continue
            if "Records" in job:      # S3 event notification: an upload with a sidecar <name>.json job file
                rec = job["Records"][0]["s3"]
                key = rec["object"]["key"]
                if not key.endswith(".json"):
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m["ReceiptHandle"]); continue
                body = boto3.client("s3").get_object(Bucket=rec["bucket"]["name"], Key=key)["Body"].read()
                job = json.loads(body)
            try:
                out = process(job)
            except Exception as e:
                out = {"job_id": job.get("job_id", "?"), "outcome": "error", "error": f"{type(e).__name__}: {e}",
                       "tb": traceback.format_exc()[-2000:]}
            ddb.put_item(Item=_ddb(out | {"created": int(time.time())}))
            cw.put_metric_data(Namespace="Gaugeless", MetricData=[
                {"MetricName": "JobSeconds", "Value": float(out.get("seconds", 0)), "Unit": "Seconds"},
                {"MetricName": "MsPerFrame", "Value": float(out.get("ms_per_frame", 0)), "Unit": "Milliseconds"},
                {"MetricName": "Outcome", "Dimensions": [{"Name": "outcome", "Value": out["outcome"]}], "Value": 1, "Unit": "Count"}])
            flood = bool((out.get("published") or {}).get("flood_alert"))
            needs_human = out["outcome"] == "escalated" or (out.get("refused") or "").startswith("flood")
            if state_machine and (needs_human or flood):
                sfn.start_execution(stateMachineArn=state_machine, name=f"{out['job_id']}-{int(time.time())}",
                                    input=json.dumps({"job_id": out["job_id"], "site": out.get("site"),
                                                      "reason": out.get("escalated") or "flood threshold",
                                                      "q": (out.get("discharge") or {}).get("q")}, default=str))
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=m["ReceiptHandle"])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--local")
    ap.add_argument("--queue-url", default=os.environ.get("GAUGELESS_QUEUE"))
    ap.add_argument("--table", default=os.environ.get("GAUGELESS_TABLE"))
    ap.add_argument("--state-machine", default=os.environ.get("GAUGELESS_SFN"))
    a = ap.parse_args()
    if a.local:
        out = process(json.loads(a.local))
        print(json.dumps({k: v for k, v in out.items() if k != "trace"}, indent=1, default=str))
    else:
        serve(a.queue_url, a.table, a.state_machine)
