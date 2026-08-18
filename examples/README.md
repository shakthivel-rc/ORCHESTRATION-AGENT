# switchboard examples

Every file here **runs**, offline, with **zero optional dependencies**, on Python
3.10+. There is no API key, no network call and no recorded-cassette machinery: each
example wires a `Router` to a small deterministic callable that stands in for a model
(plan §4.1's bring-your-own-callable path). CI executes all of them on every PR
(plan §10.4 stage 4) so they cannot rot.

```bash
pip install -e .            # or: pip install switchboard
python examples/quickstart_byo.py
```

Working from a source checkout without installing, put `src/` on the path instead:

```bash
PYTHONPATH=src python examples/quickstart_byo.py
```

---

## The files

| Example | What it shows | Extras needed |
|---|---|---|
| [`quickstart_byo.py`](quickstart_byo.py) | The plan §3.7 quickstart, **verbatim**, running end to end on a plain callable. Then: why confidence reads `0.0`, the small-catalog shortlist bypass, all four decision kinds from one router, and the audit record. | **none** |
| [`support_triage/`](support_triage/) | **The flagship.** A 126-route support catalog with entitlements and args models, plus a labelled gold set — retrieval at scale, recall@K, the entitlement gate, and a scored run. | **none** |
| [`langgraph_node.py`](langgraph_node.py) | A LangGraph conditional-edge node: `await router.aroute(...)` → `Command(goto=...)`. Replaces the `.with_structured_output()` boilerplate. | `langgraph` for the real graph; the node itself runs bare |
| [`fastapi_app.py`](fastapi_app.py) | An async `POST /chat` handler with per-tenant entitlements, clarify as a 200 and abstain as a 422. | `fastapi` for a real server; the handler runs bare |
| [`adk_agent.py`](adk_agent.py) | The router as an ordinary Google ADK tool — `pick_route(query, tool_context) -> dict`, audit record stripped. | `google-adk` for a real agent; the tool runs bare |

The three framework files guard their framework import and fall back to a few-line
stand-in, so the module still imports and `__main__` still exercises the half that
matters — the routing. That is deliberate: **switchboard has no framework
integration to install.** It returns a typed object; the "integration" is a `match`
statement you write.

---

## `quickstart_byo.py` — the zero-dependency proof

```bash
python examples/quickstart_byo.py
```

The block between the `plan §3.7 quickstart` banners is byte-for-byte what plan §3.7
prints, with exactly one substitution: `client="instructor:openai/gpt-5-nano"` →
`client=keyword_stub`. It is a lint-checkable contract — if the plan's snippet changes,
this file must change with it. (The `# noqa` pragmas on those lines exist only so the
snippet can keep the plan's import order and line width without being reformatted.)

First line of output is the snippet's own `print`:

```
track_order None 0.0
```

`0.0` is not a bug, and the example spends a paragraph on it: confidence is computed
*outside* the model from token logprobs, a bare callable returns none, so under the
no-signal rule (plan §6.3) the score is 0.0 and the clarify/abstain thresholds are
deliberately **inert** rather than acting on a number the evidence distrusts. Point
`client=` at a provider that returns logprobs and it lights up with no other change.

## `support_triage/` — the flagship catalog and benchmark fixture

```bash
python examples/support_triage/demo.py          # or: python -m examples.support_triage.demo
```

* [`catalog.py`](support_triage/catalog.py) — 126 routes over twelve support areas
  (billing, payments, shipping, returns, orders, account, technical, subscriptions,
  product, loyalty, privacy, legal) with descriptions written the way plan §5.7
  prescribes — *what it does, when to use it, when not to*. Twelve `args_model`s, ten
  routes gated behind `Route.requires`, one pinned `human_handoff` that doubles as
  `Router(fallback=...)`. It also exports `GOLD_CASES`: 71 labelled
  `(query, expected)` pairs — 61 single-route, 5 genuinely ambiguous (expect a
  `clarify`), 5 out of scope (expect an `abstain`).
* [`demo.py`](support_triage/demo.py) — runs the catalog through a `Router` and prints
  the shortlist behaviour at 126 routes (`shortlist="auto"` is above its bypass
  threshold here, so BM25 genuinely retrieves: K=10 from the plan §5.3 band table, plus
  the pinned route, in seeded-shuffle order), a scored pass over `GOLD_CASES` with
  recall@K, the entitlement gate, and three narrated decisions.

**Read `demo.STUB_NOTICE` before reading the score.** The demo's client is a keyword
matcher over trigger phrases the catalog itself declares, so its accuracy is a wiring
check, not a benchmark result — a real number needs a real model and a stated date
(plan §10.3 item 3). What *is* meaningful is everything the stub does not control:
recall@K, the entitlement filter, argument validation, the fallback and the audit trail.

## Going live

Every example is one keyword argument away from a real provider:

```python
router = Router(registry, client="instructor:openai/gpt-5-nano", ...)   # pip install 'switchboard[instructor]'
router = Router(registry, client="litellm:gemini/gemini-2.5-flash-lite", ...)  # pip install 'switchboard[litellm]'
```

Adapters are imported lazily at `Router(...)` construction, so a missing extra raises
`MissingDependencyError` there rather than on your first production request. See plan
§10.2 for the full extras matrix; a bare BYO callable needs none of it.
