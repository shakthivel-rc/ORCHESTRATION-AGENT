from __future__ import annotations

import json
from typing import TYPE_CHECKING

from switchboard import __version__
from switchboard.cli import main
from switchboard.evals import dogfood_suite, save_suite

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_version(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_eval_inspect(tmp_path: Path, capsys) -> None:
    suite = dogfood_suite(n_routes=12)
    path = save_suite(suite, tmp_path / "suite.jsonl")

    assert main(["eval", "inspect", str(path)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == suite.name
    assert payload["cases"] == len(suite.cases)
    assert payload["routes"] == len(suite.registry())


def test_cli_eval_dogfood_runs_offline(capsys) -> None:
    assert main(["eval", "dogfood", "--routes", "20", "--no-baseline", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"].startswith("dogfood")
    assert payload["candidate"]["metrics"]["n_cases"] > 0
