# Implementation Plan: Cross-Encoder Reranker and Cascaded Verifier for FedRAG

This document is written to be executed by a coding agent (or a human) task by task. Each task has a goal, exact files, interfaces, acceptance criteria and tests. Do the tasks in order. Do not start a task until the previous one's acceptance criteria pass.

## 0. Ground rules for the implementer

1. **Read before writing.** This plan was written from the README only, not from the source. Before Task 1, read these files and note how they actually work. Where this plan disagrees with the code, follow the code and adapt the plan's integration points, keeping the same behavior:
   - `coordinator/app/api/v1/endpoints.py` (query handler)
   - `coordinator/app/tasr/router.py`, `feedback.py`, `db.py`
   - `coordinator/app/flwr_server/strategy.py`
   - `coordinator/app/synthesis/aggregator.py`
   - `shared/schemas/*.py`, `shared/config.py`, `coordinator/config.yaml`
   - `nodes/app/rag/profiler.py` (how node profiles/centroids are produced and stored)
   - existing tests in `coordinator/tests/`, `tests/`, `tests/mocks/malicious_node.py`
2. **Everything new is behind config flags that default to OFF.** With all flags off, the coordinator must behave exactly as before, and all existing tests must still pass.
3. **Do not change** the Flower message format, the node-side code in `nodes/`, or the attestation/admission logic.
4. **No network calls at request time.** Models are loaded once (lazily, on first use or at startup via config) from a local cache. Document the model download step.
5. **Never block the FastAPI event loop** with model inference. Run inference with `asyncio.to_thread` (or the project's existing executor pattern).
6. **Fail safe.** If the reranker or a verifier stage raises or times out, log a warning, skip that component for the query, and fall back to the legacy ranking. Controlled by `fail_open: true` in config.
7. Follow the existing code style and tooling in `pyproject.toml` (formatter, linter, type hints). Add type hints and docstrings to all new public functions.
8. Scope is limited to what is in this document. **Out of scope:** any LLM-as-judge, adaptive or escalating defense modes, Bayesian trust changes, and any change to routing. Do not add them.

## 1. Goal and success criteria

Add two coordinator-side components:

1. **Reranker**: a cross-encoder that scores each (query, chunk) pair, replacing bi-encoder cosine similarity as the relevance signal for ranking and for TASR feedback.
2. **Cascaded verifier**: staged checks on retrieved chunks. Cheap checks run on everything; the NLI stage runs only on chunks the earlier stages could not settle.

The work is successful when:
- with flags on, retrieval quality (nDCG@k, MRR) improves over the baseline on the evaluation set;
- malicious chunks reach the final top-k less often under the attack scenarios;
- added p95 latency stays within the budget in section 9;
- with flags off, behavior and existing tests are unchanged.

## 2. Target architecture

New packages (all under `coordinator/app/`):

```
rerank/
  __init__.py
  base.py              # Reranker protocol, RerankResult
  cross_encoder.py     # CrossEncoderReranker implementation
  factory.py           # build_reranker(config) -> Reranker | None
verify/
  __init__.py
  types.py             # Candidate, Verdict, StageResult, CascadeResult
  stages.py            # IntegrityStage, OutlierStage, RelevanceGateStage, NLIStage
  nli.py               # NLIModel wrapper
  cascade.py           # Cascade runner
  node_stats.py        # per-node running statistics (Welford), persisted
  factory.py           # build_cascade(config) -> Cascade | None
```

Query path (inside the existing query handler, after node results are collected and before ranking and synthesis):

1. Collect node results as today.
2. Build `Candidate` objects (text, node_id, returned embedding, retrieval score, node trust).
3. If `rerank.enabled`: score all candidates once with the reranker (one batched call, results cached by text hash within the query).
4. If `verifier.enabled`: run the cascade, which returns a verdict, confidence and stage reached per candidate.
5. Drop `REJECT` candidates. Rank the rest (section 6).
6. Feed verdicts and rerank scores into TASR feedback (section 7).
7. Pass the top-ranked accepted evidence to synthesis as before.

## 3. Configuration schema

Add to `coordinator/config.yaml` and the config loader in `shared/config.py` (or wherever coordinator config is parsed). All keys optional with the defaults shown.

```yaml
rerank:
  enabled: false
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  backend: "sentence_transformers"   # "sentence_transformers" | "onnx"
  device: "cpu"
  batch_size: 32
  max_length: 512
  max_candidates: 50                 # cap before scoring; keep highest retrieval-score candidates
  normalization: "sigmoid"           # "sigmoid" | "minmax"
  timeout_s: 5.0

verifier:
  enabled: false
  fail_open: true
  integrity:
    enabled: true
    min_embedding_cosine: 0.90       # returned embedding vs coordinator re-embedding
    max_chars: 8000
  outlier:
    enabled: true
    z_reject: 4.0
    z_flag: 2.5
    min_samples: 30                  # do not reject on outlier until node has this many samples
  relevance_gate:
    enabled: true                    # requires rerank.enabled
    reject_below: 0.20
    accept_above: 0.80
  nli:
    enabled: false
    model: "cross-encoder/nli-deberta-v3-base"
    max_length: 384
    anchors: 3                       # trusted chunks used as reference
    contradiction_reject: 0.80
    entailment_accept: 0.70
    timeout_s: 8.0

ranking:
  mode: "legacy"                     # "legacy" | "rerank"
  # legacy: score * trust   (current behavior)
  # rerank: rerank_score * trust * verifier_confidence

trust:
  verifier_weight: 0.0               # 0.0 = verifier does not affect trust updates
  rerank_as_relevance: false         # use rerank_score as TASR relevance signal
```

Validate config at startup. Fail with a clear error if `relevance_gate.enabled` or `ranking.mode: rerank` is set while `rerank.enabled` is false.

## 4. Data types (Task 2 defines these exactly)

```python
# coordinator/app/verify/types.py
from dataclasses import dataclass, field
from enum import Enum

class Verdict(str, Enum):
    ACCEPT = "accept"        # confident good, exit cascade
    REJECT = "reject"        # confident bad, exit cascade, drop chunk
    UNCERTAIN = "uncertain"  # pass to next stage

@dataclass
class Candidate:
    idx: int                          # position in the merged candidate list
    text: str
    node_id: str
    node_trust: float
    retrieval_score: float
    returned_embedding: list[float] | None
    recomputed_embedding: list[float] | None = None   # coordinator re-embedding of text
    rerank_score: float | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class StageResult:
    verdict: Verdict
    confidence: float                 # in [0, 1]; confidence in the verdict
    stage: str                        # stage name
    reason: str = ""
    features: dict = field(default_factory=dict)

@dataclass
class CascadeResult:
    results: dict[int, StageResult]   # keyed by Candidate.idx; final result per candidate
    stage_counts: dict[str, int]      # how many candidates each stage processed
    timings_ms: dict[str, float]
```

```python
# coordinator/app/rerank/base.py
from typing import Protocol

class Reranker(Protocol):
    name: str
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one score in [0, 1] per passage, same order as input."""
```

```python
# coordinator/app/verify/stages.py
class Stage(Protocol):
    name: str
    def run(self, query: str, candidates: list[Candidate], ctx: "CascadeContext") -> dict[int, StageResult]:
        """Return a result for every candidate passed in."""
```

`CascadeContext` carries: config, node stats store, the list of all candidates (needed by NLI to pick anchors), and the node trust map.

Cascade semantics: stages run in order. Each stage receives only candidates whose current verdict is `UNCERTAIN`. A candidate leaves the cascade on `ACCEPT` or `REJECT`. A candidate that reaches the end still `UNCERTAIN` is kept, with its confidence reduced (multiply by 0.5).

## 5. Tasks

### Task 1: Baseline and evaluation fixtures

**Goal:** a repeatable way to measure before and after.

**Files to create:**
- `scripts/eval_retrieval.py`
- `tests/fixtures/eval_small.jsonl` (about 20 hand-written queries with known relevant passages, used by tests)
- `scripts/EVAL_README.md` (short usage notes)

**Eval data format** (JSONL, one object per line):

```json
{"query": "...", "relevant_texts": ["exact chunk text", "..."], "domain": "finance"}
```

**Behavior:** the script calls the running coordinator `POST /api/v1/query` with `use_llm: false`, reads `sources`, and computes nDCG@5, MRR and Recall@5 by matching returned `text` against `relevant_texts` (exact match after whitespace normalization). It also records wall-clock latency per query and reports p50 and p95. It accepts `--config-label` (a string stored in the output) and `--output results.json`.

**Acceptance:**
- running against the unmodified coordinator produces a JSON file of baseline metrics;
- unit test with a mocked HTTP client verifies the metric functions on a tiny hand-computed example.

### Task 2: Types, config and factories (no behavior change)

**Goal:** scaffolding only.

**Do:**
- create the `rerank/` and `verify/` packages with the types in section 4;
- add the config schema from section 3 with validation and defaults;
- implement `build_reranker(config)` and `build_cascade(config)` returning `None` when disabled;
- add `sentence-transformers` (and `onnxruntime` only if implementing the ONNX backend) to `pyproject.toml` / `requirements.txt`, keeping the Python constraint `>=3.10,<3.12`.

**Acceptance:**
- `pytest` for existing suites passes unchanged;
- new unit tests: config defaults load, invalid combinations raise, factories return `None` when disabled.

### Task 3: Cross-encoder reranker

**Files:** `rerank/cross_encoder.py`, `rerank/factory.py`, `coordinator/tests/test_reranker.py`.

**Implementation notes:**
- `CrossEncoderReranker.__init__(model, device, batch_size, max_length, normalization)` loads the model lazily on first `score` call and logs load time once.
- Primary backend: `sentence_transformers.CrossEncoder`. The model's raw output is a logit; apply `normalization`: `sigmoid` maps each logit independently; `minmax` normalizes within the call (document that minmax scores are not comparable across queries).
- Handle empty input (return `[]`), very long text (truncate via `max_length`), and non-string input (raise `TypeError`).
- Score cache: a small per-call dict keyed by `hash(text)` so duplicate chunks are scored once. Do not use a global cache across queries unless it is bounded (LRU, max 10,000 entries).
- Optional ONNX backend: only after everything else is done; keep it behind `backend: onnx`.

**Integration (in the query handler):**
- if `rerank.enabled`: take the merged candidates, keep at most `max_candidates` by retrieval score, call `reranker.score(query, texts)` via `asyncio.to_thread`, set `candidate.rerank_score`;
- on exception or timeout: log, leave `rerank_score = None`, and use legacy ranking for that query if `fail_open` is true.

**Tests:**
- stub model returning fixed logits: verify ordering, normalization range [0, 1], empty input, duplicate handling;
- one marked-slow integration test (`@pytest.mark.slow`) that loads the real model and checks that a clearly relevant passage scores higher than a clearly irrelevant one;
- test that a raising reranker with `fail_open` returns legacy ordering instead of an error.

**Acceptance:** tests pass; with `rerank.enabled: true` and `ranking.mode: rerank`, the query endpoint returns results ordered by the new scoring; with it off, output is byte-identical to before for a fixed fixture.

### Task 4: Response schema additions (backward compatible)

**Files:** `shared/schemas/*.py` (the source/response models), `coordinator/app/api/v1/endpoints.py`.

**Add optional fields** to each returned source: `rerank_score: float | None`, `verdict: str | None`, `verifier_confidence: float | None`, `stage_exited: str | None`. Add `routing_info.verifier` (optional dict) containing `stage_counts`, `timings_ms`, and `rejected_count`.

**Acceptance:** existing API tests pass without modification; new tests assert the new fields appear only when the corresponding feature is enabled.

### Task 5: Verifier stages 0 to 2 (cheap checks)

**Files:** `verify/stages.py`, `verify/cascade.py`, `verify/node_stats.py`, tests.

Stage order: `IntegrityStage` → `OutlierStage` → `RelevanceGateStage` → `NLIStage` (Task 6).

**IntegrityStage (name `integrity`)** returns `REJECT` (confidence 1.0) if any of:
- text is empty or whitespace only, or longer than `max_chars`;
- `returned_embedding` is missing, contains NaN/inf, or its dimension differs from the coordinator's embedding dimension;
- cosine(`returned_embedding`, `recomputed_embedding`) < `min_embedding_cosine`.

Otherwise `UNCERTAIN` with confidence 1.0. The coordinator already re-embeds returned text; reuse that result by populating `recomputed_embedding` rather than embedding twice. Put the cosine value in `features`.

**OutlierStage (name `outlier`)**
- For each node, maintain running mean and variance (Welford) of the cosine distance between a chunk's recomputed embedding and that node's profile centroid. Obtain the centroid from wherever profiles are stored (see `profiler.py` and `tasr/db.py`; confirm during the read step).
- Compute `z = (d - mean) / max(std, eps)`.
- If node sample count < `min_samples`: return `UNCERTAIN`, record `z` in features, do not reject.
- If `z > z_reject`: `REJECT`, confidence `min(1.0, z / (2 * z_reject))`.
- If `z > z_flag`: `UNCERTAIN` with `features["flagged"] = True` and confidence reduced to 0.6.
- Else `UNCERTAIN`, confidence 1.0.
- **Update the node statistics only with chunks that finish the cascade as `ACCEPT`** (or `UNCERTAIN` and not flagged), never with rejected chunks, so attackers cannot drag the baseline toward themselves. Persist statistics (JSON or the existing DB in `tasr/db.py`) so they survive restarts.

**RelevanceGateStage (name `relevance_gate`)** uses `rerank_score`:
- `< reject_below` → `REJECT` (confidence `1 - score`);
- `>= accept_above` and the candidate was not flagged by the outlier stage → `ACCEPT` (confidence `score`);
- otherwise → `UNCERTAIN` (confidence 0.5).
- If `rerank_score` is `None`, return `UNCERTAIN` for all (stage is a no-op).

**Cascade runner (`Cascade.run(query, candidates, ctx) -> CascadeResult`)**
- iterates stages in order, passing only `UNCERTAIN` candidates onward;
- records `stage_counts` and per-stage `timings_ms`;
- wraps each stage in try/except; on exception, with `fail_open`, treats the stage as a no-op and logs.

**Tests:**
- Integrity: each rejection condition with a crafted candidate; a good candidate passes.
- Outlier: with synthetic node stats, a far chunk is rejected, a near chunk passes, nothing is rejected below `min_samples`, and rejected chunks do not update stats.
- Relevance gate: three bands.
- Cascade: short-circuit behavior (a rejected chunk never reaches later stages, verified by a stage spy), stage counts, `fail_open` behavior.

**Acceptance:** all tests pass; with only these stages enabled, the `malicious_node.py` mock's garbage output is rejected at the integrity or outlier stage in `tests/e2e` style tests that run without Docker (use in-process fakes if the e2e tests need the cluster).

### Task 6: NLI stage (contradiction check against trusted evidence)

**Purpose:** catch poisoned passages that look relevant but contradict the best-supported evidence. NLI on (chunk, raw query) is poorly defined, so this stage compares uncertain chunks against high-confidence anchor chunks from other nodes.

**Files:** `verify/nli.py`, `verify/stages.py` (add `NLIStage`), tests.

**NLIModel wrapper**
- Wraps `cross-encoder/nli-deberta-v3-base` (or the configured model) via `sentence_transformers.CrossEncoder`.
- Read label order from the model config (`id2label`) instead of hardcoding it, and expose `predict(pairs) -> list[dict(contradiction, entailment, neutral)]` as softmax probabilities.
- Truncate to `max_length`.

**NLIStage (name `nli`)**
1. Anchors: choose up to `anchors` candidates from the **ACCEPT-ed** set (from earlier stages), ranked by `rerank_score * node_trust`, preferring distinct nodes. If fewer than 1 anchor exists, return `UNCERTAIN` for all (no-op).
2. For each remaining `UNCERTAIN` candidate, build pairs (anchor, candidate) skipping pairs from the same node. Truncate each text to its first `max_length` tokens.
3. Aggregate over anchors: `max_contradiction = max(p.contradiction)`, `max_entailment = max(p.entailment)`.
4. `max_contradiction >= contradiction_reject` → `REJECT` (confidence = that probability).
5. `max_entailment >= entailment_accept` → `ACCEPT` (confidence = that probability).
6. Otherwise `UNCERTAIN`.
- Cap total pairs per query (config `nli.max_pairs`, default 60) to bound latency; if exceeded, process candidates in descending `rerank_score` order and leave the rest `UNCERTAIN`.
- Run inference via `asyncio.to_thread`; honor `timeout_s` (on timeout, treat as no-op).

**Known limitation to document in code comments:** if all anchors are poisoned or the correct answer is a minority view, the stage can penalize honest chunks. Mitigations already in place: anchors come from highest rerank-and-trust candidates, and the stage only rejects at high contradiction probability.

**Tests:**
- stub NLI model returning controlled probabilities: reject on high contradiction, accept on high entailment, uncertain otherwise;
- same-node pairs skipped;
- no anchors → no-op;
- pair cap respected;
- label-order test using a fake model config with a non-default `id2label`.
- one `@pytest.mark.slow` test with the real model on a clearly contradictory pair and a clearly entailed pair.

**Acceptance:** tests pass; the stage never receives candidates already decided by earlier stages (verified by spy).

### Task 7: Ranking integration

**File:** query handler / wherever ranking is computed.

Implement `ranking.mode`:
- `legacy`: existing behavior, unchanged.
- `rerank`: `final = rerank_score * node_trust * verifier_confidence`. If `rerank_score` is `None` for a candidate (reranker failure), use the legacy score for the whole query rather than mixing scales.
- Drop all `REJECT` candidates before ranking. Never return a rejected chunk in `sources`.
- If every candidate is rejected, return an empty `sources` list, `answer: null`, and a `routing_info.verifier.all_rejected: true` flag. Do not call the LLM with empty evidence.

**Tests:** both modes on a fixed fixture; the all-rejected case; mixed rerank availability falls back to legacy.

### Task 8: TASR feedback integration

**File:** `coordinator/app/tasr/feedback.py` (and `router.py` only if trust is read there).

First read how the three existing signals (relevance, profile consistency, peer agreement) are computed and combined. Then:
- if `trust.rerank_as_relevance`: compute the relevance signal from `rerank_score` instead of cosine similarity, keeping the same range and aggregation per node;
- if `trust.verifier_weight > 0`: add a verifier term per node for the query, computed as the (confidence-weighted) accept rate of that node's candidates, minus a stronger penalty for hard `integrity` rejects. Combine it using the same update rule the existing signals use, scaled by `verifier_weight`.
- With both options off, trust updates must be identical to current behavior.

**Tests:**
- regression test: with defaults, trust update for a fixed fixture equals the pre-change values;
- a node with all chunks rejected at integrity loses trust faster than one with soft rejects;
- an honest node with all accepts does not lose trust.

### Task 9: Dashboard and observability

**Files:** `frontend/pages/1_Admin_Dashboard.py`, `coordinator/app/api/v1/endpoints.py`.

- Add `GET /api/v1/verifier/summary` returning per-node accept, reject and uncertain counts, per-stage pass-through counts, and recent stage timings (p50, p95).
- Show these in the admin dashboard as tables.
- Add structured log lines per query: stages run, counts, total verifier time.

**Acceptance:** endpoint test; dashboard renders without errors when the verifier is disabled (shows a "disabled" message).

### Task 10: Attack scenarios and evaluation

**Files:** `scripts/eval_attacks.py`, extensions to `tests/mocks/malicious_node.py` (new modes only, do not change existing behavior).

Add malicious node modes:
- `garbage`: random text with random embeddings;
- `offtopic`: valid text and embeddings from an unrelated domain;
- `mismatch`: text and embedding that do not match each other;
- `poisoned`: a passage highly similar to the query that contains a false answer (build from the eval set by replacing the answer span);
- `colluding`: several nodes return the same poisoned passage.

**Experiment matrix** (run through `eval_attacks.py`, results to JSON):
- Variants: (A) baseline, (B) reranker only, (C) reranker + stages 0 to 2, (D) reranker + stages 0 to 3.
- Conditions: 0%, 10%, 20%, 30% malicious nodes, each attack type.
- Repeat each configuration 3 times with different seeds.

**Metrics:**
- nDCG@5, MRR, Recall@5 on the eval set;
- malicious chunk rate in the final top-5;
- attack success rate (a poisoned passage appears in the top-3);
- false rejection rate on honest nodes (honest chunks rejected / honest chunks seen);
- added latency p50 and p95 per variant and per stage.

**Ablations:** reranker model choice (MiniLM vs `BAAI/bge-reranker-v2-m3`), each stage removed in turn, threshold sweeps for `reject_below`, `accept_above`, `z_reject`, `contradiction_reject`.

**Threshold tuning rule:** split the eval data into validation and test halves. Tune thresholds on validation only. Report final numbers on test only.

### Task 11 (optional): Answer-level support check

Only after Tasks 1 to 10 are done. After synthesis, split the generated answer into sentences, run NLI with premise = the accepted evidence chunks (concatenated and truncated) and hypothesis = each sentence, and return `answer_support` (fraction of sentences with entailment >= threshold) in the response. Do not regenerate or block the answer; just report the score. Behind `verifier.answer_check.enabled`, default off.

## 6. Ranking formula summary

| Mode | Formula |
|------|---------|
| `legacy` | `retrieval_similarity * node_trust` (unchanged) |
| `rerank` | `rerank_score * node_trust * verifier_confidence` |

`verifier_confidence` is 1.0 when the verifier is disabled.

## 7. Testing and verification commands

Use the project's existing test runner. Suggested commands (adjust to the real layout):

```
poetry run pytest coordinator/tests nodes/tests tests -q -m "not slow and not e2e"
poetry run pytest -q -m slow            # real-model tests, run manually
```

Mark model-loading tests with `@pytest.mark.slow` and register the marker in `pyproject.toml`. Unit tests must not download models; use stubs. Every task must leave the "not slow" suite green.

## 8. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Reranker or NLI too slow on CPU | candidate caps, `max_pairs`, batching, quantized ONNX backend, timeouts with `fail_open` |
| Relevant-looking poisoned passages fool the reranker | stages 0, 1 (outlier), and 3 (NLI contradiction vs anchors) exist for this; evaluate with `poisoned` and `colluding` attacks |
| Poisoned anchors cause honest chunks to be rejected | anchors need high rerank and trust; high contradiction threshold; measure false rejection rate |
| Outlier baseline polluted by attackers | update node stats only from accepted chunks; require `min_samples` |
| Honest nodes falsely penalized, destabilizing trust | `verifier_weight` starts at 0; tune it using the false-rejection metric |
| Threshold overfitting | validation/test split |
| Slow Docker builds from model downloads | pre-download in image build or mount a model cache volume |
| Blocking the event loop | `asyncio.to_thread` for all inference |

## 9. Performance budget (targets, adjust after measuring baseline)

- Reranker: scoring 50 candidates under 300 ms p95 on the target CPU.
- Stages 0 to 2 combined: under 20 ms p95 (no model inference).
- NLI stage: under 800 ms p95 with `max_pairs: 60`.
- Total added p95 latency with everything enabled: under 1.5 s. If exceeded, reduce `max_candidates`, `anchors`, and `max_pairs` before changing models.

## 10. Definition of done

- Tasks 1 to 10 complete (Task 11 optional), each with passing tests.
- With all new flags off, existing tests pass and API output is unchanged.
- `scripts/eval_retrieval.py` and `scripts/eval_attacks.py` reproduce every reported table.
- README updated with: new config keys, model download instructions, new response fields, new endpoint, and how to run the evaluation scripts.
- A short `docs/verifier.md` explaining the stage order, semantics of `ACCEPT`/`REJECT`/`UNCERTAIN`, and known limitations.
