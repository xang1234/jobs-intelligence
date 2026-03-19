# Counterfactual Career Delta Engine Plan v2

## Goal

Build a Counterfactual Career Delta Engine that answers:

- What single change would most improve a candidate's opportunity set?
- Which industry pivots preserve the same or a similar role while increasing reachable jobs, compensation, or market tailwinds?
- Which skills, titles, or career moves produce the best tradeoff between upside and pivot cost?
- Which of a candidate's existing skills should be swapped for a modern equivalent?

This should extend the existing profile matching, trend analysis, and search infrastructure rather than creating a parallel system.

## Product Thesis

The platform already does four valuable things:

- profile-to-job matching
- hybrid semantic and keyword retrieval
- trend analysis across roles, skills, and companies
- company similarity and related-skill discovery

The next compounding addition is not another dashboard. It is a decision engine that transforms those primitives into actionable recommendations:

- add skill `X`
- replace outdated skill `A` with modern equivalent `B`
- pivot title from `A` to `B`
- stay in the same role but move into industry `Y`
- make a bounded adjacent-role and adjacent-industry move

The system should quantify these moves with explicit before/after evidence and confidence bands instead of producing generic advice text or false-precision point estimates.

## Scope

### In scope

- Market Position baseline as a first-class output
- Counterfactual recommendations for skill addition, skill substitution, title, and industry pivots
- Same-role industry pivots
- Similar-role industry pivots
- Deterministic scoring with explicit composite formula
- Confidence metadata on all estimates
- Scenario deduplication and diversity enforcement
- Thin-market detection and messaging
- Explanation of filtered-out scenarios
- Compute budget with graceful degradation
- API support with summary/detail split
- Frontend "What If?" tab integrated into Match Lab
- Unit and integration tests

### Out of scope for v1

- LLM-generated long-form coaching
- user accounts or saved plans
- fully personalized learning-path generation
- automated resume rewriting
- multi-step optimization across long time horizons

## Existing Reuse Points

The feature should build on existing components:

- Profile matching in `src/mcf/embeddings/search_engine.py` (`match_profile()` method)
- FAISS job/skill/company indexes in `src/mcf/embeddings/index_manager.py`
- Skill extraction via `_extract_skills_from_text()` (vocabulary-based, longest-match-first)
- Related skills via `get_related_skills()` (FAISS embedding similarity + cluster fallback)
- Role, skill, company, and overview trend access in `src/mcf/database.py`
- Momentum computation via `_annotate_momentum()` (trailing-3-month-average comparison)
- API contracts in `src/api/models.py`
- HTTP route registration in `src/api/app.py` (async executor pattern)
- Profile-matching UX in `src/frontend/src/pages/MatchLabPage.tsx`
- Job categories field (99.6% coverage, 28,715 distinct category combinations across 1.2M jobs)

## Core Concept

A **delta scenario** is a synthetic change applied to a candidate profile, then rescored against the market using a shared candidate pool.

Examples:

- add `Kubernetes`
- replace `Hadoop` with `Spark`
- shift target title from `Data Analyst` to `Analytics Engineer`
- keep title family but move from `Banking and Finance` to `Healthcare / Pharmaceutical`
- move from `Operations Analyst` to `Revenue Operations` in `Information Technology`

Each scenario should produce:

- a composite score
- an estimate of jobs unlocked
- fit improvement
- salary impact
- market momentum
- diversification benefit
- pivot cost
- confidence metadata (sample size, signal strength, caveats)
- concrete before/after examples (loaded on demand)

## Scenario Types

### 1. Skill Addition

Apply one to three high-signal missing or adjacent skills to the candidate profile.

Examples:

- `Python` plus `Airflow`
- `SQL` plus `dbt`
- `Kubernetes`

Generation source: Top missing skills from baseline high-fit jobs, weighted by frequency and salary premium.

### 2. Skill Substitution

Replace an outdated or less-valued skill with its modern equivalent.

Examples:

- `jQuery` → `React`
- `Hadoop` → `Spark` or `Databricks`
- `SPSS` → `Python`
- `Waterfall` → `Agile`

Generation source: For each extracted profile skill, find related skills (same cluster or high embedding similarity) with significantly higher momentum (2× or more). Score by computing the delta in job count and salary when the old skill is swapped for the new one.

This is often the most actionable recommendation type because it has the lowest pivot cost when the candidate already has adjacent experience.

### 3. Title Pivot

Shift target title while keeping industry broad.

Examples:

- `Data Analyst` → `Analytics Engineer`
- `Software Engineer` → `Platform Engineer`

Generation source: Adjacent titles from high-fit baseline jobs where the title family differs from the candidate's current targets.

### 4. Industry Pivot, Same Role

Keep the same role family and test performance in a different industry bucket.

Examples:

- `Data Analyst` in Banking → `Data Analyst` in Logistics
- `Product Manager` in E-commerce → `Product Manager` in HealthTech

This should be a first-class scenario type because it often has lower pivot cost and higher practical value than a full career change.

Generation source: Same title-family jobs in different industry buckets from the candidate pool, filtered to buckets with meaningful job counts.

### 5. Industry Pivot, Adjacent Role

Shift both role and industry, but only within a bounded semantic distance.

Examples:

- `Operations Analyst` → `Revenue Operations` in SaaS
- `Compliance Analyst` → `Risk Analyst` in FinTech

This should carry a higher pivot cost penalty than same-role industry pivots.

Generation source: Jobs in the candidate pool where both title family and industry bucket differ, but title embedding distance is below a threshold (e.g., cosine similarity > 0.6).

## Industry Pivot Design

Industry pivots need explicit handling rather than relying only on title changes.

### Available signals

- `categories` field on jobs (99.6% coverage, comma-separated MCF-defined values)
- company identity (63K unique companies)
- title text
- skill mix

### Hybrid industry classification strategy

Use a three-layer approach for robustness:

**Layer 1 — Deterministic category mapping (handles ~80% of jobs)**

Map the ~20 primary MCF category values to a two-level taxonomy:

```
Level 1 (Sector):     Technology | Finance | Healthcare | ...
Level 2 (Sub-sector): SaaS | FinTech | HealthTech | ...
```

For single-category jobs, this is a direct lookup. For multi-category jobs (e.g., `"Banking and Finance, Information Technology"`), assign a primary bucket based on the first category and a secondary based on the second.

**Layer 2 — Company-level dominant industry (handles ambiguous multi-category jobs)**

If a company has 50 jobs and 40 map to `Banking and Finance`, assign `Banking and Finance` as the company's dominant industry. Use this to resolve ambiguous jobs from the same company.

**Layer 3 — Skill-profile affinity (available after embeddings are generated)**

For remaining unknowns, use skill embeddings to compute industry affinity. Jobs in Banking have skill profiles (SQL, Risk, Compliance) that cluster differently from HealthTech (Clinical, HIPAA, EHR).

### Two-level hierarchy rationale

A flat taxonomy of 7-10 buckets cannot distinguish "small adjacent pivot" (Banking → FinTech) from "large cross-sector pivot" (Banking → Healthcare). The two-level hierarchy gives pivot-cost a second dimension: within-sector pivots are cheaper than cross-sector pivots.

### Industry taxonomy buckets

Derived from actual MCF category distribution:

| Sector | Sub-sectors | Approx. Jobs |
|--------|-------------|-------------|
| Technology | SaaS, IT Services, Cybersecurity | 170K+ |
| Engineering | Civil, Mechanical, Electrical | 70K+ |
| F&B | Restaurants, Catering, Food Manufacturing | 56K+ |
| Construction | Building, Architecture, Interior Design | 50K+ |
| Finance | Banking, Accounting, Insurance | 48K+ |
| Logistics | Supply Chain, Warehousing, Transport | 39K+ |
| Admin & Services | Secretarial, Customer Service, General Work | 55K+ |
| Sales & Retail | Retail, E-commerce, Merchandising | 31K+ |
| Education | Teaching, Training, EdTech | 30K+ |
| Marketing & Media | PR, Advertising, Design | 34K+ |
| Human Resources | Recruitment, HR Tech, Consulting | 22K+ |
| Manufacturing | Production, Repair, Industrial | 21K+ |
| Healthcare | Pharmaceutical, Medical, Life Sciences | 19K+ |
| Sciences & R&D | Laboratory, Research, Biotech | 18K+ |
| Public Sector | Government, Non-profit | TBD |

Prefer precision over coverage. Unknowns should remain uncategorized rather than forcing noisy assignments.

### Industry-pivot rules

- same-role industry pivot requires strong title-family similarity and different industry buckets
- similar-role industry pivot requires both title similarity and industry difference, with title embedding cosine similarity > 0.6
- cross-industry moves should be rewarded when they unlock more jobs, higher compensation, or stronger momentum
- cross-industry moves should be penalized when title drift and skill-gap cost are too large
- within-sector sub-sector pivots carry lower cost than cross-sector pivots

## Architecture

### Wide-Pool + Re-Score Pattern

Do not call `match_profile()` N times. Instead, use a shared candidate pool with lightweight re-scoring.

**Why this matters:** `match_profile()` does a FAISS search for `max(limit * 30, 300)` candidates, bulk-loads job rows from SQLite, then runs a per-job scoring loop. With the single-threaded `_engine_executor`, 20 scenarios × 200-400ms each = 4-8 seconds of serialized I/O. The wide-pool pattern reduces this to one FAISS search + one bulk load + 20 cheap Python re-scoring loops (~50-100ms total for all scenarios).

**How it works:**

1. **Baseline pass**: Run FAISS with a wider candidate pool (`k = max(limit * 50, 1000)`). Fetch all job rows once. Compute baseline scores. Keep the full candidate pool in memory.

2. **Scenario scoring**: For each counterfactual scenario, re-score the existing candidate pool with modified parameters:
   - **Skill addition/substitution**: Recompute `skill_overlap` with augmented skill set (cheap set intersection on already-loaded data).
   - **Title pivot**: Re-filter by new title tokens (string match on already-loaded rows).
   - **Industry pivot**: Re-filter by industry bucket (category field already loaded).

3. **Selective FAISS re-query**: Only re-embed and re-search FAISS when the scenario fundamentally changes the query vector (e.g., a title pivot that shifts semantic meaning). Even then, intersect with the wide pool rather than scoring from scratch.

### Pre-Aggregated Market Statistics Cache

Avoid per-scenario database aggregation queries. Pre-aggregate key signals at engine load time (or on first request, then cached with TTL).

```
MarketStats:
  skill_job_counts:         dict[str, int]     # "Python" → 4521
  skill_median_salaries:    dict[str, int]     # "Python" → 96000
  skill_momentum:           dict[str, float]   # "Python" → 12.5 (%)
  title_family_job_counts:  dict[str, int]     # "data_analyst" → 1823
  title_family_median_salaries: dict[str, int]
  title_family_momentum:    dict[str, float]
  industry_job_counts:      dict[str, int]     # "Banking and Finance" → 3200
  industry_median_salaries: dict[str, int]
  industry_momentum:        dict[str, float]
  total_jobs:               int
```

This is 3-4 SQL queries at startup. Every scenario then looks up `jobs_unlocked`, `salary_delta`, and `momentum` in O(1) dictionary lookups.

### New modules

- `src/mcf/career_delta.py` — orchestrator
- `src/mcf/industry_taxonomy.py` — industry classification and title families
- `src/mcf/market_stats.py` — pre-aggregated market statistics cache

### Module responsibilities

`src/mcf/career_delta.py`

```
class CareerDeltaEngine:
  def analyze(self, request) -> CareerDeltaResponse:
    pool = self._build_candidate_pool(request)    # One FAISS search + bulk load
    baseline = self._score_baseline(pool, request) # Produces MarketPosition
    scenarios = self._generate_scenarios(pool, baseline, request)
    scenarios = self._deduplicate_and_prune(scenarios)
    scored = self._score_scenarios(pool, baseline, scenarios, budget)
    diversified = self._diversify_results(scored, request.max_scenarios)
    filtered = self._collect_filtered_reasons(scored, diversified)
    return self._build_response(baseline, diversified, filtered)
```

Responsibilities:
- build and manage the shared candidate pool
- generate candidate scenarios from baseline analysis
- deduplicate and prune overlapping scenarios
- score scenarios using the composite formula
- enforce diversity across scenario types
- enforce compute budget with early termination
- collect filtered-scenario explanations
- return structured recommendations

`src/mcf/industry_taxonomy.py`

- normalize MCF category strings into two-level industry taxonomy
- handle comma-separated multi-category values (primary + secondary assignment)
- infer company-level dominant industries from job distribution
- derive title families (lowercased, seniority-stripped, phrase-pattern-based)
- determine same-role versus adjacent-role pivots using title family comparison
- compute industry distance (within-sector = low, cross-sector = high)

`src/mcf/market_stats.py`

- pre-aggregate skill, title-family, and industry statistics from database
- compute and cache momentum, median salaries, job counts
- provide O(1) lookups for scenario scoring
- support TTL-based cache refresh

### Pre-computed columns during embed-sync

Add two columns to the `jobs` table, populated during `embed-sync`:

```sql
ALTER TABLE jobs ADD COLUMN title_family TEXT;       -- e.g. "data_analyst"
ALTER TABLE jobs ADD COLUMN industry_bucket TEXT;     -- e.g. "Banking and Finance"
```

The `embed-sync` step (which already iterates all jobs) also runs:

```
for job in new_or_updated_jobs:
  job.title_family = normalize_title(job.title)
  job.industry_bucket = classify_industry(job.categories, job.company_name)
```

This enables SQL-level aggregation (`GROUP BY industry_bucket, title_family`) which is dramatically faster than Python-level bucketing at query time. It amortizes the normalization cost across all future queries.

## Data Model

### Market Position (Baseline)

`MarketPosition` is a first-class concept, not just an implementation step. It is valuable as standalone output even without recommendations.

```
MarketPosition:
  reachable_jobs:      int                  # Jobs with fit > threshold
  mean_fit:            float                # Average fit score across reachable jobs
  top_fit:             float                # Best fit score
  median_salary_annual: Optional[int]       # Median salary of reachable jobs
  salary_range:        tuple[int, int]      # P25-P75 salary range
  top_industries:      list[IndustryShare]  # Industries of reachable jobs with counts
  top_companies:       list[str]            # Most common employers
  skill_coverage:      float                # % of market-demanded skills the candidate has
  skill_gaps:          list[SkillGap]       # Top missing skills with frequency + salary lift
  extracted_skills:    list[str]            # Skills extracted from profile
```

```
SkillGap:
  skill:                  str
  frequency_in_reachable: float             # % of reachable jobs that want this
  salary_premium:         Optional[int]     # Median salary of jobs requiring this vs. not
  momentum:               float             # Is demand for this skill growing?
```

The `skill_gaps` data directly feeds scenario generation — the top skill gaps are the skill-addition candidates.

### Request

`CareerDeltaRequest`

Fields:

- `profile_text` (20-20000 chars)
- `target_titles` (list, max 10)
- `salary_expectation_annual` optional
- `employment_type` optional
- `region` optional
- `current_industry` optional
- `delta_types` optional (list of: `skill_addition`, `skill_substitution`, `title_pivot`, `industry_same_role`, `industry_adjacent_role`)
- `max_scenarios` (1-10, default 5)

### Response

`CareerDeltaResponse`

Fields:

- `baseline` (`MarketPosition`)
- `recommendations` (list of `ScenarioSummary`)
- `filtered_scenarios` (list of `FilteredScenario`, top 2-3 that were generated but rejected)
- `search_time_ms`
- `degraded` (true if compute budget was exhausted or retrieval fell back to keywords)
- `thin_market` (true if baseline had too few matches for reliable recommendations)
- `thin_market_note` (optional explanation when thin_market is true)

### Scenario Summary (always returned)

`ScenarioSummary`

Fields:

- `id` (stable identifier for detail lookup)
- `delta_type`
- `label` (e.g., "Add Kubernetes", "Pivot to HealthTech")
- `proposed_change` (structured description of what changes)
- `score`
- `jobs_unlocked`
- `median_salary_delta`
- `demand_momentum`
- `pivot_cost`
- `confidence` (`ScenarioConfidence`)
- `one_liner` (e.g., "47 more jobs, +$8K median salary, low transition cost")

### Scenario Detail (fetched on demand)

`ScenarioDetail`

Available via `GET /api/career-delta/{scenario_id}/detail` or cached server-side with short TTL.

Fields:

- `before_examples` (list of job results from baseline)
- `after_examples` (list of job results from counterfactual)
- `supporting_signals` (trend data, related skills, similar companies)
- `skill_gap_breakdown` (specific skills needed for this move)
- `risks_or_tradeoffs` (list of strings)
- `why_this_move` (structured explanation)

### Confidence Metadata

`ScenarioConfidence`

Fields:

- `sample_size` (how many jobs informed this estimate)
- `data_freshness_days` (age of newest data point)
- `signal_strength` (`strong` if >100 jobs, `moderate` if 20-100, `weak` if <20)
- `caveats` (list of strings, e.g., "Few jobs in target industry", "Salary data sparse")

### Filtered Scenario Explanation

`FilteredScenario`

Fields:

- `label` (e.g., "Pivot to Product Manager")
- `reason` (e.g., "Title distance too high (0.82) — very different skill requirements")

Include the top 2-3 scenarios that were generated but filtered out, with a one-line explanation of why. This is cheap (already scored) and increases user trust.

## Composite Scoring Formula

### Formula

```
raw_score = (
    w_opportunity  × norm(jobs_unlocked_delta)
  + w_quality      × norm(mean_fit_delta)
  + w_salary       × norm(median_salary_delta)
  + w_momentum     × norm(demand_momentum)
  + w_diversity    × norm(industry_spread_delta)
)

pivot_discount = 1.0 - pivot_cost    # [0, 1]

score = raw_score × pivot_discount
```

### Design choices

**Multiplicative pivot cost, not additive.** A high-upside move with high cost should still score lower than a moderate-upside move with no cost. Multiplicative ensures cost scales with ambition.

**Rank-based percentile normalization** across the scenario set, not min-max. Min-max is brittle when one scenario is an outlier. Percentile ranking is stable: if you have 10 scenarios, the best gets 1.0, the worst gets 0.0, regardless of absolute magnitudes.

### Default weights (tunable)

| Weight | Value | Rationale |
|--------|-------|-----------|
| `w_opportunity` | 0.30 | Most important — are there actually more jobs? |
| `w_quality` | 0.25 | Do the new jobs fit better? |
| `w_salary` | 0.20 | Does compensation improve? |
| `w_momentum` | 0.15 | Is the market growing? |
| `w_diversity` | 0.10 | Are you less concentrated? |

### Pivot cost model

Composed of sub-costs:

```
pivot_cost = (
    0.4 × skill_gap_fraction       # What % of required skills are missing?
  + 0.3 × title_distance           # How far is the title shift? (embedding cosine distance, 0-1)
  + 0.3 × industry_distance        # same sector = 0.0, adjacent sub-sector = 0.15,
                                    # cross-sector = 0.5, very distant = 0.7
)
clamped to [0, 0.95]               # Never fully zero out a scenario
```

### Scenario-type cost adjustments

| Scenario Type | Typical pivot_cost Range | Notes |
|--------------|--------------------------|-------|
| Skill addition | 0.02 - 0.15 | Low: only skill_gap contributes |
| Skill substitution | 0.05 - 0.20 | Low: related skill, minimal relearning |
| Title pivot | 0.15 - 0.50 | Medium: title_distance dominates |
| Same-role industry pivot | 0.10 - 0.30 | Low-medium: industry_distance only |
| Adjacent-role industry pivot | 0.30 - 0.65 | High: both title and industry contribute |

## Compute Budget and Graceful Degradation

### Budget structure

```
ComputeBudget:
  max_wall_time_ms:          5000    # Hard timeout for entire request
  max_scenarios_evaluated:   20      # Cap on scenarios scored
  max_candidate_pool:        1000    # Cap on FAISS candidates
```

### Behavior

The engine checks the budget between scenario evaluations. If time is exhausted, it returns whatever scenarios have been scored so far with `degraded=True` and a note explaining truncation.

Scenarios are evaluated in priority order (cheapest and most likely to be useful first — skill additions before adjacent-role industry pivots), so early termination still returns the best results.

### Thin market detection

```
MIN_BASELINE_JOBS = 10       # Need at least this many baseline matches
MIN_SCENARIO_SIGNAL = 5      # Need at least this many jobs to justify a scenario
```

If baseline matches < `MIN_BASELINE_JOBS`, return a response with `thin_market=True`, `thin_market_note` explaining the situation, empty recommendations, and the baseline `MarketPosition` (which is still useful as a diagnostic).

For individual scenarios, skip any where the delta is computed from fewer than `MIN_SCENARIO_SIGNAL` jobs. Record this in confidence metadata.

## Scenario Deduplication and Diversity

### Deduplication pipeline

After generating candidate scenarios but before scoring:

1. **Exact dedup**: If two scenarios produce identical modified profiles (after normalization), keep one.

2. **Semantic dedup**: Compute a scenario fingerprint — the set of changes applied. If two fingerprints overlap by >70% (Jaccard similarity on changed skills/titles/industries), keep the one with lower estimated pivot cost (more actionable).

### Diversity enforcement

After scoring, ensure the top-N results include at least one representative from each scenario type that produced viable candidates. Without this, skill-additions (lowest pivot cost) will dominate every result set.

Algorithm:
1. Group scored scenarios by `delta_type`.
2. Take the best from each type that produced candidates (round-robin).
3. Fill remaining slots by global score rank.
4. Sort final list by score descending.

## API Plan

### Endpoints

Add two endpoints:

- `POST /api/career-delta` — main analysis, returns baseline + scenario summaries
- `GET /api/career-delta/{scenario_id}/detail` — detail for a specific scenario (cache lookup)

### Routing

Register endpoints in `src/api/app.py` using the existing async executor pattern with `_engine_executor`.

### Detail caching

The main endpoint computes full details for all scenarios but returns only summaries. Full details are stored in a server-side TTL cache (5-minute TTL, keyed by scenario ID). The detail endpoint is just a cache lookup — no recomputation.

### Validation

Add new request and response models to `src/api/models.py`.

Validation should enforce:

- minimum profile length (20 chars)
- bounded scenario counts (1-10)
- supported delta types
- reasonable title list sizes (max 10)
- `salary_expectation_annual` must be positive if provided

## Frontend Plan

### Integration point

Add a **"What If?" tab** within the existing Match Lab page, not a separate page. The career delta shares the same input fields (profile text, target titles, salary, region) and produces a complementary output.

```
┌─────────────────────────────────────────────────────┐
│  Match Lab                                           │
│                                                      │
│  [Profile Input]          [Results Panel]            │
│  ┌──────────────┐        ┌────────────────────────┐ │
│  │ paste resume │        │ [Matches] [What If?]   │ │
│  │ ...          │        │                        │ │
│  │ target title │        │  (tab content here)    │ │
│  │ salary       │        │                        │ │
│  │ region       │        └────────────────────────┘ │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

### Rationale for integration over separate page

1. **Shared input**: Avoids duplicating the form or making the user paste their profile twice.
2. **Complementary flow**: "Here are your matches" and "here's how to get better matches" are two sides of the same coin.
3. **Baseline reuse**: The match results are the baseline. Same-page placement avoids redundant computation.
4. **Lower activation energy**: A tab surfaces the feature exactly when it's relevant, vs. requiring discovery of a separate page.

### "What If?" tab sections

- Baseline summary (Market Position card: reachable jobs, median salary, skill coverage, top industries)
- Skill gaps (top missing skills with frequency and salary premium)
- Ranked recommendation cards (scenario summaries)
- Filtered scenarios (collapsed section: "We also considered..." with reasons for exclusion)
- Confidence indicators (visual: solid bar for strong signal, dotted for weak)

### Recommendation card contents

- Scenario label and delta type badge
- Jobs unlocked count
- Salary delta (with confidence indicator)
- Pivot cost gauge (low/medium/high visual)
- One-liner summary
- "Explore" button to load detail

### Expanded scenario detail (on click)

- Before/after job examples (side by side)
- Skill gap breakdown for this specific move
- Supporting trend data
- Risk/tradeoff notes

### "Apply this scenario" interaction

When viewing a scenario's detail, an "Apply" button pre-fills the Match Lab input with the modified profile/title, so the user can immediately see the changed match results. This creates a natural exploration loop: see matches → see recommendations → apply one → see improved matches.

### Loading and error states

- Loading spinner with "Analyzing your market position..." message
- Thin market state: informative message with suggestion to broaden inputs
- Error state: retry button
- Empty scenario state: "Your profile is well-positioned — no high-confidence improvements found"

## Testing Plan

### Unit tests

Add deterministic tests for:

- industry normalization (single category, multi-category, unknown)
- two-level taxonomy mapping (sector + sub-sector)
- company-dominant industry inference
- title-family normalization (seniority stripping, noise removal, family grouping)
- same-role cross-industry detection
- adjacent-role detection (title embedding distance threshold)
- scenario generation rules (correct scenario types from given baseline)
- composite score formula (explicit weight verification)
- pivot cost model (per-scenario-type cost ranges)
- rank-based percentile normalization
- scenario deduplication (exact and semantic)
- diversity enforcement (type representation)
- thin-market detection threshold
- compute budget early termination
- confidence metadata assignment (sample size → signal strength mapping)
- skill substitution detection (momentum comparison)

### Integration tests

Use the existing database factory style to build small controlled markets and test:

- skill-addition scenarios produce correct job-count deltas
- skill-substitution scenarios detect outdated → modern skill pairs
- title pivots produce correct fit deltas
- same-role industry pivots correctly filter by industry bucket
- adjacent-role industry pivots respect title distance threshold
- candidate pool reuse produces same results as full re-runs (correctness check)
- market stats cache matches fresh database queries
- degraded-mode behavior (keyword fallback)
- thin-market response shape
- diversity enforcement with skewed scenario distribution

### API tests

Add endpoint tests for:

- valid response shape (baseline + summaries + filtered)
- scenario detail endpoint returns cached data
- scenario detail endpoint returns 404 after TTL expiry
- capped scenario count respected
- no-result profiles return thin_market response
- sparse-category markets handled gracefully
- degraded retrieval fallback
- request validation (min profile length, max titles, valid delta types)
- compute budget timeout returns partial results with degraded flag

### Frontend tests

Add integration-level UI tests:

- submit profile and receive "What If?" tab with recommendation cards
- baseline summary displays correct metrics
- expand a scenario and inspect before/after evidence
- "Apply" pre-fills Match Lab with modified inputs
- thin-market state shows informative message
- loading state appears during analysis
- confidence indicators render correctly for strong/moderate/weak signals

## Phased Delivery

### Phase 1 — Engine Core + All Scenario Types

- `career_delta.py` orchestrator with candidate pool reuse
- `industry_taxonomy.py` (deterministic category mapping — it's a small module)
- `market_stats.py` (pre-aggregated statistics cache)
- `MarketPosition` baseline as first-class output
- All five scenario types: skill addition, skill substitution, title pivot, same-role industry pivot, adjacent-role industry pivot
- Composite scoring with explicit formula and multiplicative pivot cost
- Scenario deduplication and diversity enforcement
- Thin-market detection
- Compute budget with early termination
- Confidence metadata on all estimates
- Unit tests for all normalizers, scoring, deduplication, and budget logic

### Phase 2 — API + Integration Tests

- `CareerDeltaRequest` and `CareerDeltaResponse` models in `src/api/models.py`
- `POST /api/career-delta` endpoint
- `GET /api/career-delta/{scenario_id}/detail` endpoint with TTL cache
- Filtered scenario explanations in response
- Integration tests with controlled market fixtures
- API validation and error handling tests

### Phase 3 — Frontend: "What If?" Tab

- "What If?" tab in MatchLabPage
- Baseline summary (Market Position card)
- Ranked recommendation cards with confidence indicators
- Loading states, error handling, thin-market state, empty state

### Phase 4 — Frontend: Expansion + Polish

- Scenario detail expansion (before/after job examples)
- "Apply this scenario" flow (pre-fill MatchLab inputs)
- Skill gap breakdown visualization
- Filtered scenarios section ("We also considered...")
- Frontend integration tests

### Phase 5 — Pre-computed Columns + Tuning

- Add `title_family` and `industry_bucket` columns to jobs table
- Populate during `embed-sync`
- Weight tuning based on user testing
- Latency optimization (SQL-level aggregation on pre-computed columns)
- Broader regression coverage

## MVP Acceptance Criteria

- Given a realistic profile, the system returns three to five ranked recommendations with diversity across scenario types.
- At least one recommendation is an industry pivot for the same or a very similar role.
- Each recommendation includes explicit evidence (jobs unlocked, salary delta, pivot cost) with confidence metadata (signal strength, sample size).
- The baseline Market Position is a standalone useful output even when recommendations are empty.
- The system returns filtered scenario explanations for the top 2-3 rejected moves.
- The endpoint works in degraded mode using deterministic fallback retrieval.
- The endpoint respects compute budget and returns partial results gracefully under timeout.
- Thin markets produce an informative response rather than empty or misleading recommendations.
- Recommendations are stable under test fixtures and do not depend on free-form generation.
- Scenario diversity enforcement prevents any single scenario type from monopolizing all result slots.

## Key Design Principles

- Prefer deterministic recommendation logic over free-form advice generation.
- Reuse existing retrieval and trend infrastructure via candidate pool reuse, not N redundant searches.
- Keep industry pivots first-class rather than hiding them inside title pivots.
- Optimize for explainability, testability, and product compounding value.
- Penalize unrealistic pivots multiplicatively so that cost scales with ambition.
- Be honest about uncertainty — use confidence bands, not false-precision point estimates.
- Explain exclusions as well as inclusions to build user trust.
- Pre-aggregate and pre-compute where possible — amortize normalization cost at sync time, not query time.
- Enforce diversity in results — the user needs strategic breadth, not five variations of "add Python."

## Recommended First Slice

Implement the smallest useful slice in this order:

1. industry taxonomy module (deterministic category mapping)
2. title-family normalizer
3. market statistics cache
4. baseline Market Position computation
5. skill-addition scenarios
6. skill-substitution scenarios
7. title pivots
8. same-role industry pivots
9. adjacent-role industry pivots
10. composite scoring with deduplication and diversity
11. compute budget and thin-market handling
12. API endpoints (summary + detail)
13. frontend "What If?" tab with recommendation cards
14. scenario detail expansion and "Apply" flow

This gets a useful engine into the product quickly while preserving a clean path toward richer recommendations and deeper frontend exploration.
