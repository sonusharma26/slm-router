"""Synthetic multi-model routing fixture ("model zoo"). NOT public benchmark evidence.

Twelve simulated models span roughly two orders of magnitude in price. They differ in
general skill, domain specialties and reasoning ability. Each request belongs to one
of eight domains, has a latent difficulty and receives a binary correctness score per
model. Request text is generated from domain and difficulty vocabularies so lexical
features carry real but imperfect signal. Use it to exercise routing code paths; real
claims require external datasets such as LLMRouterBench.
"""
from __future__ import annotations

import math
import random

from inference_control.benchmarks.data import BenchmarkDataset, Query, Truth, grouped_split
from inference_control.contracts import EndpointSnapshot, RequestContext
from inference_control.routing.features import NGRAM_FEATURE_VERSION, ngram_features
from inference_control.util import digest

DOMAINS = {
    "math": ["integral", "derivative", "equation", "polynomial", "matrix", "eigenvalue", "probability", "triangle", "prime", "theorem", "series", "limit"],
    "code": ["python", "function", "array", "recursion", "compile", "bug", "algorithm", "complexity", "regex", "api", "thread", "pointer"],
    "law": ["contract", "statute", "liability", "plaintiff", "tort", "jurisdiction", "precedent", "clause", "appeal", "negligence"],
    "medicine": ["diagnosis", "symptom", "dosage", "patient", "clinical", "therapy", "chronic", "infection", "cardiac", "lesion"],
    "history": ["empire", "treaty", "revolution", "dynasty", "colonial", "medieval", "war", "century", "monarchy", "republic"],
    "chemistry": ["molecule", "reaction", "acid", "bond", "oxidation", "catalyst", "isotope", "compound", "solvent", "enthalpy"],
    "writing": ["essay", "poem", "story", "paragraph", "tone", "narrative", "headline", "rewrite", "persuasive", "metaphor"],
    "logic": ["puzzle", "deduce", "riddle", "syllogism", "premise", "constraint", "grid", "knights", "liars", "sequence"],
}
GENERIC = ["please", "explain", "the", "following", "question", "answer", "carefully", "given", "consider", "what", "how", "which", "find", "show"]
HARD = ["prove", "rigorous", "derive", "optimize", "formal", "multistep", "edge", "cases", "counterexample", "nested"]
EASY = ["simple", "basic", "quick", "short", "define", "list", "name"]

# name, input $/M, output $/M, skill, specialties, reasoning, base latency ms
MODELS = (
    ("tiny-1b", .04, .08, -1.4, {}, False, 120),
    ("small-8b", .08, .16, -.6, {}, False, 180),
    ("coder-14b", .15, .3, -.6, {"code": 2.0, "logic": .5}, False, 220),
    ("math-7b", .1, .2, -.9, {"math": 2.4, "logic": .9}, False, 200),
    ("med-13b", .15, .3, -.6, {"medicine": 2.0, "chemistry": .8}, False, 220),
    ("legal-13b", .15, .3, -.6, {"law": 2.0, "history": .7}, False, 220),
    ("writer-24b", .25, .5, -.1, {"writing": 1.6}, False, 260),
    ("mid-32b", .4, .8, .4, {}, False, 320),
    ("reason-32b", .5, 2., .5, {"math": .6, "logic": .8}, True, 900),
    ("large-70b", .9, 2.7, 1.0, {}, False, 450),
    ("frontier", 3., 15., 1.6, {}, False, 700),
    ("frontier-reason", 5., 25., 1.8, {"math": .5, "code": .4, "logic": .6}, True, 2200),
)


def zoo_endpoints() -> list[EndpointSnapshot]:
    return [EndpointSnapshot(endpoint_id=name, provider="zoo", upstream_model=name, revision="zoo-r1",
        region="local", context_window=131072, config_hash="zoo-config", price_version="zoo-price-1",
        input_price_per_million=price_in, output_price_per_million=price_out, governance=frozenset({"local"}),
        capabilities=frozenset({"structured_output", "tools"}))
        for name, price_in, price_out, *_ in MODELS]


def _unit(*parts) -> float:
    return int(digest(parts)[:16], 16) / 2**64


def make_zoo(n: int = 4000, seed: int = 7, *, task_hints: bool = False) -> BenchmarkDataset:
    """Generate a deterministic full-information matrix of n requests x 12 models."""
    if n < 200:
        raise ValueError("at least 200 requests required")
    rng = random.Random(seed)
    names = sorted(DOMAINS)
    endpoints = zoo_endpoints()
    queries, outcomes = [], []
    for i in range(n):
        domain = names[rng.randrange(len(names))]
        second = names[rng.randrange(len(names))] if rng.random() < .15 else None
        difficulty = min(1., max(0., rng.betavariate(2., 2.)))
        words = rng.sample(DOMAINS[domain], k=rng.randint(3, 6))
        if second:
            words += rng.sample(DOMAINS[second], k=2)
        words += rng.sample(HARD, k=min(3, int(difficulty * 4))) if difficulty > .55 else rng.sample(EASY, k=rng.randint(0, 2))
        words += rng.sample(GENERIC, k=rng.randint(3, 6))
        rng.shuffle(words)
        text = f"{' '.join(words)} #{i}"
        group = digest((seed, text))
        input_tokens = int(len(words) * 1.4) + 40
        request = RequestContext(request_id=f"zoo-{seed}-{i}", application_id="benchmark", tenant_policy_id="benchmark",
            input_tokens=input_tokens, max_output_tokens=16384, feature_version=NGRAM_FEATURE_VERSION,
            task_hint=domain if task_hints else None, traffic_slices=frozenset({domain}) if task_hints else frozenset(),
            query_features=ngram_features(text))
        queries.append(Query(request=request, text=text, split=grouped_split(group, seed), group_id=group))
        for name, price_in, price_out, skill, special, reasoning, latency in MODELS:
            ability = skill + special.get(domain, 0.) + (1.2 * difficulty if reasoning else 0.)
            logit = 1.3 * (ability + 1.6 - 3.4 * difficulty)
            p = 1 / (1 + math.exp(-logit))
            correct = float(_unit(seed, i, name, "correct") < p)
            output = int((180 + 700 * difficulty) * (4.5 if reasoning else 1.) * (.8 + .4 * _unit(seed, i, name, "len")))
            cost = (input_tokens * price_in + output * price_out) / 1e6
            ms = latency * (1 + difficulty) * (.85 + .3 * _unit(seed, i, name, "lat"))
            outcomes.append(Truth(request_id=request.request_id, endpoint_id=name, quality=correct, cost=cost,
                latency_ms=ms, input_tokens=input_tokens, output_tokens=output, evaluator_version="zoo-binary-v1"))
    return BenchmarkDataset(name=f"synthetic-model-zoo-{seed}", evidence_kind="synthetic", queries=queries,
        outcomes=outcomes, endpoints=endpoints, provenance={"seed": seed, "task_hints": task_hints,
        "purpose": "routing code-path fixture with domain specialists; not empirical evidence",
        "quality": "simulated binary correctness from a latent skill model"})


FREE_FORM = frozenset({"writing"})  # outputs that never match exactly, like real essays


def zoo_answers(dataset: BenchmarkDataset) -> dict[tuple[str, str], str]:
    """Simulated extracted answers for the zoo (synthetic; for cascade code paths only).

    A correct model gives the reference answer. Wrong models share mistakes: half the time
    the common wrong answer, otherwise one of three others, so two wrong models agree about
    a third of the time. Free-form domains get unique outputs that never agree.
    """
    seed = dataset.provenance.get("seed", 7)
    domains = {}
    for query in dataset.queries:
        words = set(query.text.split())
        domains[query.request.request_id] = max(DOMAINS, key=lambda d: (len(words & set(DOMAINS[d])), d))
    answers = {}
    for row in dataset.outcomes:
        key = (row.request_id, row.endpoint_id)
        if domains[row.request_id] in FREE_FORM:
            answers[key] = f"free-form:{row.endpoint_id}:{row.request_id}"
        elif row.quality >= 1:
            answers[key] = "reference"
        else:
            draw = _unit(seed, row.request_id, row.endpoint_id, "answer")
            answers[key] = "common-mistake" if draw < .5 else f"mistake-{int((draw - .5) * 6)}"
    return answers
