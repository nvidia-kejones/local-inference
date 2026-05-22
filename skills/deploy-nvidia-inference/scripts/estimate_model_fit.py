#!/usr/bin/env python3
"""Estimate approximate model/runtime fit for one host and workload."""

from __future__ import annotations

import argparse

from common_io import load_structured, write_json
from fitlib import estimate_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="normalized host_facts.json")
    parser.add_argument("--workload", required=True, help="workload profile YAML or JSON")
    parser.add_argument("--candidate", required=True, help="candidate JSON or YAML")
    parser.add_argument("--out", help="write fit report JSON here")
    args = parser.parse_args()

    candidate = load_structured(args.candidate)
    if isinstance(candidate, dict) and "candidates" in candidate:
        raise SystemExit("--candidate must contain one candidate, not a candidate set")
    report = estimate_fit(load_structured(args.host), load_structured(args.workload), candidate)
    write_json(report, args.out)


if __name__ == "__main__":
    main()
