I would make the next version a **proof release**—call it something like **SLM Router v0.3: Evidence & Validation**. Freeze almost all new functionality until we can answer:

> **Given the same models, requests and information available at decision time, does our planner produce a better feasible inference strategy than existing routers?**

That is especially important now because the bar has moved. Current vLLM Semantic Router is itself a programmable mixture-of-models layer targeting quality, cost, latency, privacy and safety—not just a basic semantic classifier—and its v0.4 release landed in September 2026. [GitHub](https://github.com/vllm-project/semantic-router?utm_source=chatgpt.com)

And LLMRouterBench is almost perfect for keeping us intellectually honest: 400K+ instances, 21+ datasets, 33 models and 10 routing baselines. More importantly, its authors found that several sophisticated routers **do not reliably beat simple baselines under unified evaluation**. [arXiv](https://arxiv.org/abs/2601.07206?utm_source=chatgpt.com)

So here's how I would proceed.

---

# 1. Build the benchmark harness first

Before changing the routing algorithm, establish a fixed experiment.

Your existing design already calls for a full-information matrix:

\[
\text{request}\times\text{endpoint}\rightarrow\text{observed result}
\]

and records quality, tokens, cost, latency, revisions, provenance, etc. Pasted markdown

Keep that.

### Benchmark layer A — standard router benchmark

Start with **LLMRouterBench**.

Initially restrict SLM Router to **direct routing only**.

That deliberately removes our fancy cascades and parallel execution. We're answering:

> Is our evidence/calibration machinery itself competitive as a model selector?

Run the exact same model pool against:

| Router | Purpose |
|---|---|
| Fixed-small | Cost floor |
| Fixed-best | Quality ceiling |
| Random | Sanity baseline |
| Cheapest feasible | Simple constraint baseline |
| Threshold router | Strong simple baseline |
| kNN / embedding router | Semantic baseline |
| RouteLLM | Established learned router |
| GraphRouter / strongest LLMRouterBench methods | Current research baseline |
| SLM Router direct | Ours |
| Oracle | Upper bound |

RouteLLM already provides serving/evaluation infrastructure and claims major cost reduction at fixed performance, so it is a legitimate baseline rather than a strawman. [GitHub](https://github.com/sha367/routellm?utm_source=chatgpt.com)

### Benchmark layer B — our actual advantage

Then create an **extended inference-plan benchmark** supporting:

\[
\text{direct}
\]

\[
\text{cascade}
\]

\[
\text{verify}\rightarrow\text{escalate}
\]

\[
\text{parallel}\rightarrow\text{select}
\]

Existing routing benchmarks mostly evaluate:

\[
x\rightarrow M_i
\]

Our important claim is:

\[
x\rightarrow\pi_i
\]

where \(\pi_i\) is an **execution plan**.

That's where SLM Router should attempt to differentiate.

---

# 2. Fix calibration before optimizing routing

Right now I wouldn't train a smarter router.

I'd build a **calibration layer around every prediction**.

For endpoint \(e\) and request \(x\), suppose some model predicts:

\[
\hat q(x,e)
\]

for quality.

We shouldn't let the planner consume \(\hat q\).

It should consume something closer to:

\[
L_q(x,e)
\]

where \(L_q\) is a calibrated lower confidence/prediction bound.

Likewise:

\[
U_L(x,e)
\]

for latency and:

\[
U_C(x,e)
\]

for uncertain cost.

Then feasibility becomes:

\[
L_q \ge Q_{\min}
\]

\[
U_L\le L_{\max}
\]

\[
E[C]\le C_{\text{expected}}
\]

\[
U_C\le C_{\max}
\]

plus privacy/capability/provider constraints.

### Start with one-sided conformal calibration

For quality, train any predictor:

\[
\hat q_i=f(x_i,e_i)
\]

On the completely separate calibration set calculate residuals:

\[
r_i=\hat q_i-q_i
\]

and derive a one-sided conformal quantile \(R_\alpha\).

Then:

\[
\boxed{L_q(x,e)=\hat q(x,e)-R_\alpha}
\]

Do the inverse for latency:

\[
\boxed{U_L(x,e)=\hat L(x,e)+R_\beta}
\]

preferably over log-latency because latency distributions are strongly skewed.

Recent routing work is already exploring conformal risk control specifically for cost-aware routing, so this is a credible direction rather than something we'd be inventing without precedent. [arXiv](https://arxiv.org/abs/2605.12001?utm_source=chatgpt.com)

But we should be very precise about the guarantee: basic conformal coverage is generally a **distributional/marginal guarantee**, not magical certainty that a specific individual request will succeed.

### Make calibration request-conditioned

Use something analogous to Mondrian/grouped calibration for:

```text
endpoint
× task family
× application
× token-length bucket
× traffic slice
× privacy class
× structured/tool requirement
× model revision
```

This fits the evidence system you've already implemented. Pasted markdown

If there isn't enough evidence:

**abstain rather than fabricate precision.**

That is exactly where your existing `ESTIMATE_MISSING` philosophy becomes valuable.

### Measure calibration separately from routing

A router getting 95% accuracy tells us nothing about whether its claimed 95% confidence is meaningful.

Track:

| Calibration metric | Why |
|---|---|
| Coverage | Does the bound actually cover at its claimed rate? |
| ECE | Are probabilities calibrated? |
| Brier score | Probability quality |
| Interval width | Are certificates useful or trivially conservative? |
| Worst-slice coverage | Hidden failure modes |
| Certification rate | How often can we actually route? |
| Abstention rate | Price paid for certainty |

---

# 3. Solve compound-plan evidence correctly

This may become one of the best parts of the project.

Suppose:

```text
Small → verifier → frontier if verifier rejects
```

A dangerous shortcut would be:

\[
P(\text{success})
=
P(S)\times P(V)\times P(F)
\]

because those outcomes are **not independent**.

Easy requests make both the SLM and verifier look good. Hard requests can make both fail together.

So don't infer compound reliability from marginal endpoint evidence.

### Build a joint evidence matrix

For every benchmark request retain:

```text
request
    small-model output
    medium-model output
    frontier output

    verifier(small-output)
    verifier(medium-output)

    evaluator(small-output)
    evaluator(medium-output)
    evaluator(frontier-output)
```

Once those artifacts exist, thousands of plans can be **replayed offline without paying for inference again**.

For every plan \(\pi\), calculate:

\[
Q_\pi
\]

\[
C_\pi
\]

\[
L_\pi
\]

\[
N_{\text{calls},\pi}
\]

\[
\text{escalation}_\pi
\]

\[
\text{failure}_\pi
\]

from the actual correlated outcomes.

Now calibrate:

\[
L_Q(x,\pi)
\]

directly.

This turns:

> “we think this cascade should be reliable”

into:

> “we have observed this exact execution strategy on comparable requests.”

### Canonical plan identity

Every compound strategy needs a fingerprint incorporating:

```text
endpoint revisions
order
verifier revision
decision thresholds
selection algorithm
prompt/schema/tool hashes
timeout rules
fallback rules
```

Change any materially relevant component → old joint certificate cannot silently survive.

That fits extremely well with your existing certificate/invalidation architecture.

---

# 4. Replace the drift detector with a safety state machine

The current failure—**8 latency violations in 40 decisions with no detected drift**—is probably the most important bug in the project right now. Pasted markdown

Don't solve this with one smarter anomaly detector.

Split drift into three fundamentally different classes.

### Deterministic drift

Things such as:

```text
price changed
model revision changed
endpoint disappeared
schema/capability changed
```

Need **zero statistical detection**.

Immediately invalidate affected certificates.

### Fast observable drift

Things we know immediately:

```text
latency
timeouts
HTTP errors
token usage
actual spend
availability
```

Monitor residuals:

\[
r_t=L_t-\hat L_t
\]

rather than raw latency.

Use something simple and auditable such as **CUSUM/EWMA** plus hard tripwires.

Example:

```text
NORMAL
   ↓
SUSPECT
   ↓
QUARANTINED
   ↓
PROBING
   ↓
RECERTIFIED
   ↓
NORMAL
```

A statistical detector should not be the only defense.

If a supposedly ≤500 ms endpoint violates 500 ms repeatedly, a hard circuit-breaker should quarantine it even if the fancy drift statistic hasn't accumulated enough evidence.

### Slow/delayed quality drift

Quality is harder because outcomes may arrive minutes, hours or days later.

Maintain sequential evidence over trusted outcomes.

A confidence-sequence/e-process style test is especially interesting here because evidence is arriving continuously and we may inspect it repeatedly.

When evidence becomes insufficient:

\[
\text{ACTIVE}\rightarrow\text{UNCERTIFIED}
\]

not:

\[
\text{ACTIVE}\rightarrow\text{hope}
\]

Then active measurement spends part of its probe budget trying to recertify it.

---

# 5. Turn drift into a benchmark, not just unit tests

You already identified the right scenarios:

price changes, endpoint disappearance, model revision, 3× latency, quality regression, workload shift, delayed outcomes and corrupted evaluators. Pasted markdown

For each injection record:

\[
T_{\text{detect}}
\]

\[
N_{\text{violations-before-detection}}
\]

\[
\$ _{\text{damage-before-detection}}
\]

\[
T_{\text{recovery}}
\]

\[
\$ _{\text{probe}}
\]

\[
FP_{\text{rate}}
\]

This gives us a much stronger graph than merely saying “drift detection works.”

For example:

```text
              Violations before quarantine
SLM Router                1.3
EWMA only                 4.8
Static router            17.6
No adaptation            23.1
```

That would be valuable evidence.

---

# 6. Make the evaluation brutally leakage-safe

Your existing protocol is already correct here:

**training → calibration → policy-selection validation → blind test → temporal/model holdout → leave-domain-out.** Pasted markdown

Keep it.

Most importantly:

The router cannot see anything that would not exist when an actual production request arrives.

So absolutely no:

```text
realized output length
actual latency
future outcome
ground-truth evaluator result
other endpoint's answer
```

as routing features.

Prompt tokens, task metadata, declared requirements and historical endpoint evidence are legitimate.

Keep all paraphrases/variants of the same logical problem together in one split as your protocol already requires. Pasted markdown

---

# 7. Define victory before running the blind test

This is important because otherwise we will unconsciously move the goalposts.

I would define the release decision something like this:

| Requirement | Target |
|---|---:|
| Privacy/capability violations | **0** |
| Duplicate external dispatch | **0** |
| Replay mismatch | **0** |
| Calibration coverage | At nominal target within statistical uncertainty |
| Cost vs strongest router | **≥10% lower at matched quality**, preferably CI excluding 0 |
| Quality vs strongest router | No material regression at matched cost |
| Worst-slice quality | Meets policy |
| Severe injected drift | Detected before predefined violation budget |
| Probe/verification costs | Included in total cost |
| Blind test | Never touched during development |

And report **Pareto curves**, not one headline average.

The project's existing acceptance criteria already correctly say to compare cost at matched quality, quality at matched cost, worst-slice performance, abstention, calibration, total probe+serving spend and drift recovery—not merely average router accuracy. Pasted markdown

---

# Implementation order

I would do exactly these six changes, in this order:

1. **`benchmark/`** — LLMRouterBench adapter, common baseline interface, immutable splits and reproducible result artifact.
2. **`calibration/`** — one-sided calibrated quality/latency/cost bounds + calibration reports.
3. **`evidence/compound/`** — plan fingerprints, joint plan observations and offline plan replay.
4. **`monitoring/drift/`** — NORMAL/SUSPECT/QUARANTINED/PROBING state machine + sequential detectors + hard breakers.
5. **`benchmarks/adapters/`** — RouteLLM/current vLLM Semantic Router/strong LLMRouterBench baselines under identical model pools.
6. **`evaluation/blind_release.py`** — one command that opens the untouched test split, computes bootstrap confidence intervals/Pareto frontiers and generates the final release report.

And **do not add another routing feature until #6 has run**.

---

## The most important research question

After all of this, I would actually stop asking:

> “Does our router predict the best model better?”

The deeper hypothesis is:

\[
\boxed{
\text{Can calibrated execution-plan search dominate model routing
when real deployment constraints matter?}
}
\]

LLMRouterBench already suggests that simply inventing another classifier-based router may have diminishing returns—the oracle gap remains significant and many methods converge toward similar performance. [arXiv](https://arxiv.org/abs/2601.07206?utm_source=chatgpt.com)

Our opportunity is different:

**model routing → inference planning**

and

**predicted best model → cheapest certifiably feasible execution plan**.

If that wins experimentally, *that* is the result that could move SLM Router from an impressive engineering project to something genuinely research-worthy.