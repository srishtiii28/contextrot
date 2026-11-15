"""Reliability-layer evaluation CLI. Run from context-rot-backend/.

  python scripts/eval_reliability.py run      --detector judge --data sessions.jsonl
  python scripts/eval_reliability.py fit      --detector judge [--fit-weights]
  python scripts/eval_reliability.py evaluate --detector judge [--split test]
  python scripts/eval_reliability.py compare  --v1 judge --v2 nli

`run` caches per-turn scores so `fit` and `evaluate` never call the model again. `fit` uses only the dev
split. `evaluate --split test` scores the held-out set once with frozen thresholds and refuses to run
again unless --force (slide 9 validation protocol).
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reliability import evaluation  # noqa: E402

REPORTS = Path("reports")


def _scores_path(detector):
    return REPORTS / f"reliability-scores-{detector}.jsonl"


def _thresholds_path(detector):
    return REPORTS / f"reliability-thresholds-{detector}.json"


def _read_rows(detector):
    path = _scores_path(detector)
    if not path.exists():
        sys.exit(f"{path} not found: run `run --detector {detector}` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


def cmd_run(args):
    sessions = evaluation.load_sessions(args.data)
    rows = asyncio.run(evaluation.run_detector(sessions, args.detector))
    REPORTS.mkdir(exist_ok=True)
    meta = dict(detector=args.detector, data=str(args.data), data_sha256=evaluation.data_fingerprint(args.data))
    _scores_path(args.detector).write_text("\n".join(json.dumps(dict(r, **meta)) for r in rows) + "\n")
    print(f"Scored {len(rows)} turns from {len(sessions)} sessions -> {_scores_path(args.detector)}")


def cmd_fit(args):
    rows = _read_rows(args.detector)
    result = evaluation.fit(rows, fit_weights=args.fit_weights)
    result.update(detector=args.detector, fitted_at=datetime.now(UTC).isoformat(),
                  data_sha256=rows[0].get("data_sha256"),
                  dev_report=evaluation.report(evaluation.rescore([r for r in rows if r["split"] == "dev"], result["weights"]), result["tau"]))
    _write_json(_thresholds_path(args.detector), result)
    print(f"\nSet RECOVERY_THRESHOLD={result['tau']} and W_* weights from this file to apply it.")


def cmd_evaluate(args):
    rows = [r for r in _read_rows(args.detector) if r["split"] == args.split]
    if not rows:
        sys.exit(f"No {args.split}-split rows")
    thresholds_path = _thresholds_path(args.detector)
    if not thresholds_path.exists():
        sys.exit(f"{thresholds_path} not found: fit on dev first; thresholds are never fitted on test")
    frozen = json.loads(thresholds_path.read_text())
    marker = REPORTS / f"reliability-heldout-scored-{args.detector}.json"
    if args.split == "test" and marker.exists() and not args.force:
        sys.exit(f"Held-out set already scored ({marker}). It is scored once; use --force only to reproduce that run.")
    result = evaluation.report(evaluation.rescore(rows, frozen["weights"]), frozen["tau"])
    result.update(detector=args.detector, split=args.split, thresholds=str(thresholds_path),
                  evaluated_at=datetime.now(UTC).isoformat())
    _write_json(REPORTS / f"reliability-eval-{args.detector}-{args.split}.json", result)
    if args.split == "test":
        marker.write_text(json.dumps(dict(evaluated_at=result["evaluated_at"], tau=frozen["tau"])))


def cmd_compare(args):
    reports = []
    for detector in (args.v1, args.v2):
        path = REPORTS / f"reliability-eval-{detector}-{args.split}.json"
        if not path.exists():
            sys.exit(f"{path} not found: evaluate {detector} first")
        reports.append(json.loads(path.read_text()))
    _write_json(REPORTS / f"reliability-compare-{args.split}.json", evaluation.compare(*reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run"); p.add_argument("--detector", required=True, choices=["rules", "judge", "nli"]); p.add_argument("--data", required=True, type=Path); p.set_defaults(func=cmd_run)
    p = sub.add_parser("fit"); p.add_argument("--detector", required=True); p.add_argument("--fit-weights", action="store_true"); p.set_defaults(func=cmd_fit)
    p = sub.add_parser("evaluate"); p.add_argument("--detector", required=True); p.add_argument("--split", default="test", choices=["dev", "test"]); p.add_argument("--force", action="store_true"); p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("compare"); p.add_argument("--v1", default="judge"); p.add_argument("--v2", default="nli"); p.add_argument("--split", default="test", choices=["dev", "test"]); p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
