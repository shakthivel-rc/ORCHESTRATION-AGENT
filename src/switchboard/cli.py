"""Small stdlib CLI over the APIs that already ship.

The CLI intentionally starts narrow: version reporting and an offline dogfood eval
that needs no provider key. Heavy report rendering, YAML gates, enrichment and
training commands remain future work rather than hidden imports of optional
packages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from switchboard import ClientCapabilities, LLMRequest, LLMResult, Router, Usage, __version__
from switchboard.evals import (
    BASELINES,
    EvalCase,
    ExpectedClarify,
    ExpectedMultiRoute,
    ExpectedRoute,
    dogfood_suite,
    load_suite,
    run_suite,
)

_QUERY_RE = re.compile(r"<user_request>\n(.*?)\n</user_request>", re.DOTALL)


class _GoldClient:
    """Deterministic offline client for CLI smoke/eval runs."""

    capabilities = ClientCapabilities()
    model_id = "byo:gold-fixture"
    provider = "fixture"

    def __init__(self, cases: list[EvalCase]) -> None:
        self._cases = {case.query: case for case in cases}

    def complete(self, request: LLMRequest) -> LLMResult[Any]:
        query = _query_from_request(request)
        case = self._cases.get(query)
        payload = _payload_for(case) if case is not None else {
            "rationale": "No fixture case matched the query.",
            "kind": "abstain",
            "reason": "model_elected",
        }
        return LLMResult(
            parsed=None,
            raw_text=json.dumps(payload, ensure_ascii=False),
            usage=Usage(),
            model_id=self.model_id,
            provider_meta={"adapter": "fixture"},
        )


def _query_from_request(request: LLMRequest) -> str:
    for segment in reversed(request.segments):
        found = _QUERY_RE.search(segment.content)
        if found:
            return found.group(1)
    return ""


def _payload_for(case: EvalCase) -> dict[str, Any]:
    expected = case.expected
    if isinstance(expected, ExpectedRoute):
        route = expected.any_of[0]
        return {
            "rationale": f"Fixture labels this query as {route}.",
            "kind": "route",
            "route": route,
            "args": expected.args or None,
        }
    if isinstance(expected, ExpectedMultiRoute):
        return {
            "rationale": "Fixture labels this query as multiple independent routes.",
            "kind": "multi_route",
            "routes": [{"route": route, "args": None} for route in expected.routes],
        }
    if isinstance(expected, ExpectedClarify):
        return {
            "rationale": "Fixture labels this query as ambiguous.",
            "kind": "clarify",
            "question": "Could you clarify what you want to do?",
            "candidates": expected.acceptable_routes or None,
            "missing": expected.missing or None,
        }
    return {
        "rationale": "Fixture labels this query as out of scope.",
        "kind": "abstain",
        "reason": "model_elected",
    }


def _cmd_version(args: argparse.Namespace) -> int:
    del args
    print(__version__)
    return 0


def _cmd_eval_dogfood(args: argparse.Namespace) -> int:
    suite = dogfood_suite(n_routes=args.routes)
    registry = suite.registry()
    if registry is None:  # pragma: no cover - dogfood_suite always pins one
        raise RuntimeError("dogfood suite did not include a registry")
    router = Router(
        registry,
        client=_GoldClient(suite.cases),
        shortlist="auto",
        allow_clarify=True,
        multi_route=True,
        otel=False,
    )
    result = run_suite(
        router,
        suite,
        baselines=() if args.no_baseline else BASELINES,
        seed=args.seed,
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.summary_table())
    router.close()
    return 0


def _cmd_eval_inspect(args: argparse.Namespace) -> int:
    suite = load_suite(args.path)
    registry = suite.registry()
    payload = {
        "name": suite.name,
        "cases": len(suite.cases),
        "has_catalog": registry is not None,
        "routes": len(registry) if registry is not None else None,
        "registry_version": registry.version if registry is not None else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switchboard")
    parser.set_defaults(func=lambda _args: 0)
    sub = parser.add_subparsers(dest="command")

    version = sub.add_parser("version", help="print the installed switchboard version")
    version.set_defaults(func=_cmd_version)

    eval_parser = sub.add_parser("eval", help="evaluation helpers")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    dogfood = eval_sub.add_parser("dogfood", help="run the offline dogfood suite")
    dogfood.add_argument("--routes", type=int, default=120, help="number of dogfood routes to generate")
    dogfood.add_argument("--seed", type=int, default=None, help="provider seed stamped on baseline runs")
    dogfood.add_argument("--no-baseline", action="store_true", help="skip the naive-full-catalog baseline")
    dogfood.add_argument("--json", action="store_true", help="emit SuiteResult JSON instead of a table")
    dogfood.set_defaults(func=_cmd_eval_dogfood)

    inspect_cmd = eval_sub.add_parser("inspect", help="print metadata for an eval suite JSONL file")
    inspect_cmd.add_argument("path", help="path to a suite JSONL file")
    inspect_cmd.set_defaults(func=_cmd_eval_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
