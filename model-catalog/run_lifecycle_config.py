#!/usr/bin/env python3
"""Run one saved lifecycle request with a verified backend image."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortex_training.client.core import DEBUG_OPTIONS_ENV
from cortex_training.connection import resolve_connection
from job_lifecycle_suite import LifecycleSuite


def resolve_file_connection(path):
    """Keep ambient profiles and credentials from overriding the explicit file."""
    keys = {
        "CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_CONFIG",
        "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_HOST", "CORTEX_TRAINING_PAT",
        "CORTEX_TRAINING_DATABASE", "CORTEX_TRAINING_SCHEMA", "CORTEX_TRAINING_ENDPOINT",
        "SNOWFLAKE_HOST", "SNOWFLAKE_PAT", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
    }
    previous = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        return resolve_connection(config=str(path))
    finally:
        os.environ.update(previous)


def pin_request(request, image_tag):
    pinned = deepcopy(request)
    pinned.setdefault("debug", {}).setdefault("job", {})["image_tag"] = image_tag
    return pinned


def resolved_image_tag(image_tag):
    # The pairing manifest currently contains only Hopper hardware.
    return image_tag + "-h"


class ImageVerifiedClient:
    """Check asynchronous placement metadata before lifecycle operations."""

    def __init__(self, client, image_tag, output):
        self.client = client
        self.image_tag = image_tag
        self.output = output
        self.audit = {
            "expected_image_tag": image_tag,
            "expected_resolved_image_tag": resolved_image_tag(image_tag),
            "jobs": [],
        }

    def __getattr__(self, name):
        return getattr(self.client, name)

    def wait_for_job(self, job_id, *args, **kwargs):
        job = self.client.get_job(job_id)
        if job.get("image_tag"):
            self._verify(job_id, job)
        try:
            ready = self.client.wait_for_job(job_id, *args, **kwargs)
        except Exception as readiness_error:
            try:
                self._verify(job_id, self.client.get_job(job_id), require_match=False)
            except Exception as audit_error:
                raise readiness_error from audit_error
            raise
        self._verify(job_id, self.client.get_job(job_id))
        return ready

    def _verify(self, job_id, job, *, require_match=True):
        actual = job.get("image_tag")
        verified = actual == resolved_image_tag(self.image_tag)
        self.audit["jobs"].append({
            "job_id": job_id, "image_tag": actual, "verified": verified,
        })
        path = self.output / "image-audit.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.audit, indent=2) + "\n")
        temporary.replace(path)
        if require_match and not verified:
            raise RuntimeError(
                f"Job image mismatch: expected {resolved_image_tag(self.image_tag)!r}, "
                f"got {actual!r}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--connection-config", type=Path, required=True)
    parser.add_argument("--connection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    args = parser.parse_args(argv)
    payload = args.request.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.request_sha256:
        parser.error("saved request fingerprint changed; no job submitted")
    request = json.loads(payload)
    if request.get("debug", {}).get("job", {}).get("image_tag") != args.image_tag:
        parser.error("saved request does not match --image-tag")
    if hashlib.sha256(args.connection_config.read_bytes()).hexdigest() != args.connection_sha256:
        parser.error("connection fingerprint changed; no job submitted")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ[DEBUG_OPTIONS_ENV] = "1"
    with resolve_file_connection(args.connection_config).client as client:
        LifecycleSuite(
            ImageVerifiedClient(client, args.image_tag, args.output_dir),
            request, args.output_dir,
        ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
