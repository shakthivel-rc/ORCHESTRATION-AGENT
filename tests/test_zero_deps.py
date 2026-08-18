"""THE core guarantee: Pydantic-only core, structurally enforced (plan §2.4).

Plan §2.4 does not ask for a zero-dependency core as an aspiration; it specifies
two **blocking** CI guards, and this module is both of them plus the static
contract they back up:

1. *bare-venv test* — install with only Pydantic, run a route with a BYO
   callable, assert ``sys.modules`` contains nothing from the deny-list;
2. *import-linter contract* — forbid ``core | engine | router -> adapters/extras``.

The rule this enforces (§2.4, import topology): optional third-party imports are
permitted **only** in ``providers/*_adapter.py``, ``providers/__init__.py``,
``shortlisters/embedding_backends.py`` and ``telemetry/otel.py`` — and even there
they must be lazy (inside a function, and/or guarded by ``try/except
ImportError``), never at module top level.

Why it is worth a test file of its own: CVE-2026-42208 was an *unbounded
transitive pin* that let a credential-exfiltrating wheel resolve into installs of
a library most users never configured. Lazy imports mean a compromised optional
dependency cannot execute for a user who never installed it — but only for as
long as nothing quietly adds a module-level ``import litellm``. Nothing else in
the suite would notice that; this file is what notices.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import switchboard

# --------------------------------------------------------------------------- #
# The contract, spelled out (plan §2.4).
# --------------------------------------------------------------------------- #

PACKAGE_ROOT = Path(switchboard.__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent

DENIED_PACKAGES: frozenset[str] = frozenset(
    {
        "anthropic",
        "boto3",
        "fastembed",
        "instructor",
        "litellm",
        "numpy",
        "openai",
        "opentelemetry",
        "rich",
        "sentence_transformers",
        "sklearn",
        "torch",
        "yaml",
    }
)
"""Every optional dependency plus every heavyweight transitive one the plan names.

``numpy`` and ``torch`` are on the list even though nothing declares them
directly: they arrive through ``[embed]`` and ``[distill-train]``, and a stray
``import numpy`` in the shortlister would make the "pure Python, no numpy"
promise of §5.2 quietly false while every test still passed."""

LAZY_IMPORT_MODULES: frozenset[str] = frozenset(
    {
        "providers/__init__.py",
        "shortlisters/embedding_backends.py",
        "telemetry/otel.py",
    }
)
"""Non-adapter modules permitted a *lazy* optional import (plan §2.4).

``providers/*_adapter.py`` is allowed too and is matched by pattern below."""


def _is_adapter(relative: Path) -> bool:
    """``providers/<something>_adapter.py`` — the third permitted location."""
    parts = relative.as_posix()
    return parts.startswith("providers/") and parts.endswith("_adapter.py")


def _may_import_optional(relative: Path) -> bool:
    return relative.as_posix() in LAZY_IMPORT_MODULES or _is_adapter(relative)


def _source_files() -> list[Path]:
    """Every shipped ``.py`` under ``src/switchboard``."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _module_id(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


# --------------------------------------------------------------------------- #
# AST walk
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImportSite:
    """One import of a denied package, with the context that decides its legality."""

    module: str
    """Root package name, e.g. ``"litellm"``."""
    file: str
    line: int
    in_function: bool
    """Inside a ``def``/``async def`` body at any nesting depth."""
    import_guarded: bool
    """Inside the ``try`` block of a ``try/except ImportError`` (or bare ``except``)."""

    @property
    def lazy(self) -> bool:
        """Plan §2.4: "lazy (inside a function/method) or guarded by try/except"."""
        return self.in_function or self.import_guarded

    def __str__(self) -> str:
        return f"{self.file}:{self.line} imports {self.module!r}"


_TRY_NODES: tuple[type[ast.AST], ...] = tuple(
    node for node in (ast.Try, getattr(ast, "TryStar", None)) if node is not None
)


def _catches_import_error(node: ast.Try) -> bool:
    """Whether ``node``'s handlers include ``ImportError`` (or a bare ``except``)."""
    for handler in node.handlers:
        if handler.type is None:
            return True
        rendered = ast.unparse(handler.type)
        if "ImportError" in rendered or "ModuleNotFoundError" in rendered:
            return True
    return False


def _imported_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    """Root package names an import statement pulls in.

    Relative imports (``from .base import ...``) are skipped: they are always
    first-party and can never name an optional dependency.
    """
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return []
        return [(node.module or "").split(".", 1)[0]]
    return [alias.name.split(".", 1)[0] for alias in node.names]


def _walk(node: ast.AST, *, in_function: bool, guarded: bool, out: list[ImportSite], file: str) -> None:
    """Collect denied imports with their (function, try-guard) context."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for root in _imported_roots(node):
            if root in DENIED_PACKAGES:
                out.append(
                    ImportSite(
                        module=root,
                        file=file,
                        line=node.lineno,
                        in_function=in_function,
                        import_guarded=guarded,
                    )
                )
        return

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for child in ast.iter_child_nodes(node):
            _walk(child, in_function=True, guarded=guarded, out=out, file=file)
        return

    if isinstance(node, _TRY_NODES):
        body_guarded = guarded or _catches_import_error(node)  # type: ignore[arg-type]
        for child in node.body:  # type: ignore[attr-defined]
            _walk(child, in_function=in_function, guarded=body_guarded, out=out, file=file)
        for handler in node.handlers:  # type: ignore[attr-defined]
            for child in handler.body:
                _walk(child, in_function=in_function, guarded=guarded, out=out, file=file)
        for child in (*node.orelse, *node.finalbody):  # type: ignore[attr-defined]
            _walk(child, in_function=in_function, guarded=guarded, out=out, file=file)
        return

    for child in ast.iter_child_nodes(node):
        _walk(child, in_function=in_function, guarded=guarded, out=out, file=file)


def _scan(source: str, file: str = "<memory>") -> list[ImportSite]:
    sites: list[ImportSite] = []
    _walk(
        ast.parse(source, filename=file),
        in_function=False,
        guarded=False,
        out=sites,
        file=file,
    )
    return sites


def _denied_imports(path: Path) -> list[ImportSite]:
    return _scan(path.read_text(encoding="utf-8"), _module_id(path))


# --------------------------------------------------------------------------- #
# The walker checks itself before it checks the tree.
# --------------------------------------------------------------------------- #

_VIOLATIONS = {
    "module scope": "import litellm\n",
    "aliased": "import numpy as np\n",
    "from-import": "from openai import OpenAI\n",
    "submodule": "import opentelemetry.trace\n",
    "inside a class body": "class A:\n    import yaml\n",
    "nested in a conditional": "if True:\n    import boto3\n",
    "inside a with block": "with open('f') as fh:\n    import torch\n",
    "buried three deep": (
        "def outer():\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            for _ in range(3):\n"
        "                import fastembed\n"
    ),
}


@pytest.mark.parametrize("source", _VIOLATIONS.values(), ids=list(_VIOLATIONS))
def test_the_walker_finds_denied_imports_at_any_level(source: str) -> None:
    """A guard nobody has seen fail is not a guard (plan §2.4: "structural")."""
    assert _scan(source) != []


@pytest.mark.parametrize(
    ("source", "expected_lazy"),
    [
        ("import litellm\n", False),
        ("def f():\n    import litellm\n", True),
        ("try:\n    import litellm\nexcept ImportError:\n    litellm = None\n", True),
        (
            "def f():\n    try:\n        import litellm\n    except ImportError:\n        raise\n",
            True,
        ),
        # The *handler* is not a lazy context: an import there still runs eagerly.
        ("try:\n    pass\nexcept ImportError:\n    import litellm\n", False),
    ],
    ids=["eager", "in-function", "guarded", "both", "in-handler"],
)
def test_the_walker_classifies_laziness_correctly(source: str, expected_lazy: bool) -> None:
    """"Lazy" means "inside a function/method, or guarded by try/except" (§2.4)."""
    sites = _scan(source)

    assert len(sites) == 1
    assert sites[0].lazy is expected_lazy


def test_first_party_and_relative_imports_are_never_flagged() -> None:
    """The deny-list is about optional *third-party* packages only."""
    assert _scan("from switchboard.core.route import Route\n") == []
    assert _scan("from .base import LLMResult\n") == []
    assert _scan("import json, hashlib\nfrom pydantic import BaseModel\n") == []


# --------------------------------------------------------------------------- #
# Static contract (plan §2.4 guard 2: the import-linter contract)
# --------------------------------------------------------------------------- #


def test_the_source_tree_is_non_empty_and_parseable() -> None:
    """Guard the guard: a walker that finds no files proves nothing."""
    files = _source_files()

    assert len(files) > 20
    assert any(_module_id(path) == "router.py" for path in files)
    assert any(_is_adapter(path.relative_to(PACKAGE_ROOT)) for path in files)


@pytest.mark.parametrize(
    "path", _source_files(), ids=lambda path: _module_id(path)
)
def test_no_optional_dependency_outside_the_permitted_modules(path: Path) -> None:
    """``core/``, ``engine/``, ``errors.py``, ``router.py`` — and everything else
    that is not one of the four permitted locations — may never import an
    optional SDK, at **any** nesting level (plan §2.4).

    "At any level" is the part that matters. A module-level import is obvious in
    review; an ``import litellm`` buried in a helper three functions deep inside
    ``engine/loop.py`` is not, and it would break the bare-venv install for every
    user of the BYO-callable path.
    """
    relative = path.relative_to(PACKAGE_ROOT)
    if _may_import_optional(relative):
        pytest.skip(f"{relative.as_posix()} is a permitted optional-import location")

    offenders = _denied_imports(path)

    assert offenders == [], "\n".join(str(site) for site in offenders)


@pytest.mark.parametrize(
    "path",
    [
        path
        for path in _source_files()
        if _may_import_optional(path.relative_to(PACKAGE_ROOT))
    ],
    ids=lambda path: _module_id(path),
)
def test_permitted_modules_import_their_sdk_lazily(path: Path) -> None:
    """Even where an SDK import is allowed, it must not run at module import.

    ``providers.resolve_client`` imports adapter modules eagerly at ``Router(...)``
    construction, so an adapter module that imported its SDK at the top would
    drag ``litellm`` into a process that only ever asked for ``instructor`` —
    and ``telemetry/otel.py`` is imported by ``router.py`` unconditionally, so a
    top-level ``opentelemetry`` import there would make the ``[otel]`` extra a
    hard dependency by accident.
    """
    sites = _denied_imports(path)

    eager = [site for site in sites if not site.lazy]
    assert eager == [], "\n".join(f"{site} at module scope" for site in eager)


def test_every_shipped_adapter_is_covered_by_the_lazy_rule() -> None:
    """The permitted list is a whitelist, so it must not silently grow stale."""
    permitted = {
        _module_id(path)
        for path in _source_files()
        if _may_import_optional(path.relative_to(PACKAGE_ROOT))
    }

    assert "providers/__init__.py" in permitted
    assert "telemetry/otel.py" in permitted
    assert "shortlisters/embedding_backends.py" in permitted
    assert {"providers/instructor_adapter.py", "providers/litellm_adapter.py"} <= permitted
    # The BYO-callable adapter needs no SDK at all and must stay clean.
    assert _denied_imports(PACKAGE_ROOT / "providers" / "callable_adapter.py") == []


def test_the_public_reexport_module_imports_no_adapter() -> None:
    """"``__init__.py`` never imports any adapter" (plan §2.4, §2.3)."""
    tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))

    imported = {
        module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module in (
            [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else [alias.name for alias in node.names]
        )
    }

    assert not any("_adapter" in module for module in imported)
    assert not any(module.split(".", 1)[0] in DENIED_PACKAGES for module in imported)


# --------------------------------------------------------------------------- #
# Runtime contract (plan §2.4 guard 1: the bare-venv test)
# --------------------------------------------------------------------------- #


def _run_probe(script: str) -> str:
    """Run ``script`` in a fresh interpreter that can import switchboard."""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(SRC_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


_DENY_LIST_LITERAL = repr(sorted(DENIED_PACKAGES))

_IMPORT_PROBE = f"""
import sys
import switchboard
denied = sorted(set({_DENY_LIST_LITERAL}) & set(sys.modules))
print(",".join(denied))
"""


def test_importing_switchboard_pulls_in_no_optional_dependency() -> None:
    """``import switchboard`` in a bare venv is Pydantic + stdlib, full stop (§2.3).

    Checked in a **subprocess**: this test session has already imported half the
    library, so ``sys.modules`` in-process proves nothing about what a fresh
    ``import switchboard`` costs.
    """
    assert _run_probe(_IMPORT_PROBE) == ""


_PUBLIC_API_PROBE = f"""
import sys
import switchboard
missing = [name for name in switchboard.__all__ if not hasattr(switchboard, name)]
denied = sorted(set({_DENY_LIST_LITERAL}) & set(sys.modules))
print("|".join([",".join(missing), ",".join(denied), str(len(switchboard.__all__))]))
"""


def test_the_whole_public_api_is_reachable_with_only_pydantic() -> None:
    """Every name in ``__all__`` resolves without an extra installed (plan §2.3).

    The public surface is the promise; if reaching one corner of it needed
    ``litellm`` on the import path, the zero-dependency claim would hold only for
    the parts nobody used.
    """
    missing, denied, count = _run_probe(_PUBLIC_API_PROBE).split("|")

    assert missing == ""
    assert denied == ""
    assert int(count) >= 40


_ROUTING_PROBE = f"""
import json
import sys

from pydantic import BaseModel

from switchboard import Registry, RequestContext, Route, Router

class RefundArgs(BaseModel):
    order_id: str

registry = Registry([
    Route(name="refund", description="Issue or check a refund for an order.",
          args_model=RefundArgs, examples=("I want my money back for order 123",)),
    Route(name="track_order", description="Track shipment status for an existing order."),
    Route(name="human_handoff", description="Escalate to a human support agent.", pinned=True),
])

def byo(prompt: str) -> str:
    # The plan §4.1 BYO row: rendered prompt in, raw string out.
    return json.dumps({{
        "rationale": "the request names an order and asks for money back",
        "kind": "route",
        "route": "refund",
        "args": {{"order_id": "A-123"}},
    }})

router = Router(registry, client=byo, shortlist="auto", fallback="human_handoff", otel=False)
decision = router.route("I want my money back for order A-123",
                        context=RequestContext(tenant_id="acme"))

denied = sorted(set({_DENY_LIST_LITERAL}) & set(sys.modules))
print("|".join([decision.kind, decision.route, decision.args.order_id,
                decision.audit.registry_version, ",".join(denied)]))
"""


def test_a_full_routing_call_with_a_byo_callable_stays_clean() -> None:
    """PLAN §2.4 CI GUARD 1, verbatim.

    "Install with only Pydantic, run a route with a BYO callable, assert
    ``sys.modules`` contains nothing from a deny-list." This is the guard that
    actually protects users: the static walk above proves no *source line*
    imports an SDK, and this proves no *code path* a real decision takes reaches
    for one either — including the shortlister, the wire schema, the validator,
    the confidence ladder, the policy stage and the audit emit.
    """
    kind, route, order_id, registry_version, denied = _run_probe(_ROUTING_PROBE).split("|")

    assert kind == "route"
    assert route == "refund"
    assert order_id == "A-123"
    assert len(registry_version) == 12
    assert denied == ""


_OTEL_PROBE = f"""
import sys

from switchboard.telemetry.otel import OTelEmitter, otel_available

emitter = OTelEmitter(enabled=True)
with emitter.decision_span() as span:
    span_ok = span is not None

denied = sorted(set({_DENY_LIST_LITERAL}) & set(sys.modules))
print("|".join([str(otel_available()), str(emitter.available), str(span_ok),
                ",".join(denied)]))
"""


def test_the_otel_emitter_degrades_to_a_no_op_without_the_extra() -> None:
    """"With ``opentelemetry`` not installed this module still imports cleanly...
    every span context manager degrades to a no-op" (plan §8.1).

    Nothing in the routing loop changes shape — which is why ``router.py`` can
    import the emitter unconditionally without making ``[otel]`` a hard dep.
    """
    available, emitter_available, span_ok, denied = _run_probe(_OTEL_PROBE).split("|")

    assert available == "False"
    assert emitter_available == "False"
    assert span_ok == "True"
    assert denied == ""
