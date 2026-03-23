# Career Pathfinder Engine — Multi-Step Career Strategy Optimization

## Context

The career delta engine (completed March 17) is the project's crown jewel — it answers "what single change would most improve my position?" with 5 scenario types, composite scoring, pivot costs, and confidence metadata.

But users don't make single moves. They make sequences. The real question is: **"What's my optimal career strategy over the next 2-3 years?"**

The Career Pathfinder transforms single-step career delta into a multi-step strategy optimizer. It chains career delta evaluations through a beam search, where each step's output becomes the next step's input. This is "Google Maps for careers" — nobody else has the data (1.2M jobs, 7 years), taxonomy, embeddings, AND scenario scoring engine needed to build it.

## Core Concept

A **career path** is a sequence of career delta scenarios applied to a profile, where each step:
- Modifies the profile (add skill, change title, shift industry)
- Is scored against the real market (via `CareerDeltaEngine.analyze()`)
- Has evidence (actual jobs at that waypoint)
- Has cumulative metrics (total salary gain, total jobs unlocked, total pivot cost)

The pathfinder explores this space with **bounded beam search**:
1. Run career delta on current profile → top K scenarios (depth 0)
2. For each of top B scenarios, construct modified profile → run career delta again (depth 1)
3. Repeat to max depth D (default 3)
4. Score complete paths by cumulative value / cumulative cost
5. Return top N paths with evidence at each waypoint

## Architecture

### New Module: `src/mcf/career_pathfinder.py`

```
CareerPathfinder
├── find_paths(request, goal?) → list[CareerPath]
│   ├── Level 0: engine.analyze(current_profile) → scenarios
│   ├── Level 1: for top B scenarios → modify profile → engine.analyze() → scenarios
│   ├── Level 2: for top B scenarios → modify profile → engine.analyze() → scenarios
│   └── Prune, score, rank complete paths
├── apply_scenario(request, scenario) → CareerDeltaRequest  [construct modified profile]
├── score_path(path) → PathScore  [cumulative scoring]
└── detect_diminishing_returns(path) → bool  [pruning]
```

### Key Data Structures

```python
@dataclass(frozen=True)
class CareerPathStep:
    """One move in a career path."""
    depth: int                          # 0, 1, 2, ...
    scenario: ScenarioSummary           # The career delta scenario at this step
    baseline: BaselineMarketPosition    # Market position AFTER applying this step
    cumulative_salary_delta_pct: float  # Total salary change from start
    cumulative_jobs_unlocked: int       # Total new jobs accessible
    cumulative_pivot_cost: float        # Accumulated effort/risk
    evidence_jobs: int                  # Jobs matching at this waypoint

@dataclass(frozen=True)
class CareerPath:
    """A complete multi-step career strategy."""
    steps: tuple[CareerPathStep, ...]
    path_score: float                   # Composite score for the whole path
    origin_baseline: BaselineMarketPosition  # Starting position
    final_baseline: BaselineMarketPosition   # Ending position
    total_salary_delta_pct: float
    total_jobs_unlocked: int
    total_pivot_cost: float
    confidence: float                   # Min confidence across all steps
    path_id: str                        # Stable cache key
```

### Profile Mutation Logic: `apply_scenario()`

This is the critical bridge — given a `CareerDeltaRequest` and a `ScenarioSummary`, produce a new request reflecting the move:

| Scenario Type | Profile Mutation |
|---------------|-----------------|
| Skill Addition | Add skill to `current_skills` |
| Skill Substitution | Remove old skill, add new skill in `current_skills` |
| Title Pivot | Change `target_titles` to new title family |
| Same-Role Industry Pivot | Update `current_categories` to target industry |
| Adjacent-Role Industry Pivot | Change `target_titles` AND `current_categories` |

The `ScenarioChange` dataclass already carries `source_title_family`, `target_title_family`, `source_industry`, `target_industry` — everything needed.

### Beam Search Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_depth` | 3 | Most career strategies are 2-4 moves |
| `beam_width` | 3 | Top 3 scenarios explored per level |
| `max_paths` | 5 | Return top 5 complete paths |
| `min_improvement_pct` | 5% | Prune steps with < 5% marginal gain |
| `max_cumulative_pivot_cost` | 0.85 | Stop if path becomes unrealistically costly |

### Performance Budget

- Each `engine.analyze()` call: ~100ms (wide-pool optimization)
- Beam width 3, depth 3: worst case 3 + 9 + 27 = 39 calls (but pruning reduces this)
- Realistic: ~12-18 calls after pruning = 1.2-1.8 seconds
- Add path scoring overhead: ~2-3 seconds total
- Acceptable for an async endpoint with loading UX

### Path Scoring Formula

```
path_score = (
    0.35 * norm(total_salary_delta_pct)
    + 0.25 * norm(total_jobs_unlocked)
    + 0.20 * norm(min_step_confidence)
    + 0.15 * norm(1 - total_pivot_cost)
    + 0.05 * norm(path_efficiency)       # value per step (shorter paths preferred)
)
```

Multiplicative diminishing returns penalty: each additional step after 2 gets a 0.85 discount factor.

### Pruning Rules

1. **Diminishing returns**: Skip step if marginal salary gain < 5% AND marginal jobs < 10%
2. **Cumulative cost ceiling**: Stop path if total pivot cost > 0.85
3. **Confidence floor**: Skip step if scenario confidence < 0.30
4. **Redundancy**: Don't add a skill that a later title pivot would make irrelevant
5. **Cycle detection**: Never revisit a title_family + industry combination

## Critical Files to Modify

| File | Change |
|------|--------|
| `src/mcf/career_pathfinder.py` | **NEW** — Pathfinder engine (~400-500 lines) |
| `src/mcf/career_delta.py` | Minor — export `ScenarioChange`, ensure `analyze()` accepts synthetic profiles cleanly |
| `src/mcf/career_delta_retrieval.py` | Minor — ensure `build_candidate_pool()` works with modified skill sets |
| `src/api/models.py` | Add `CareerPathRequest`, `CareerPathResponse` API models |
| `src/api/app.py` | Add `POST /api/career-path` endpoint |
| `src/cli.py` | Add `career-path` CLI command for testing |
| `tests/test_career_pathfinder.py` | **NEW** — unit + integration tests |

### Existing Functions to Reuse

- `CareerDeltaEngine.analyze()` — core evaluation at each step (`career_delta.py:389`)
- `CareerDeltaRequest` — profile representation, already has all needed fields (`career_delta.py:56`)
- `ScenarioChange` — carries mutation info (title family, industry) (`career_delta.py:~260`)
- `BaselineMarketPosition` — market position snapshot at each waypoint (`career_delta.py:~280`)
- `rank_and_filter_scenarios()` — reuse ranking logic per step (`career_delta.py`)
- `normalize_title_family()` — title normalization (`industry_taxonomy.py`)
- `industry_distance()` — pivot cost inputs (`industry_taxonomy.py`)
- `build_scenario_id()` — stable IDs for caching (`career_delta.py:472`)
- `_salary_lift_pct()` — salary comparison util (`career_delta.py`)

## Implementation Sequence

### Phase 1: Core Pathfinder Engine (~2 days)
1. Create `src/mcf/career_pathfinder.py` with `CareerPathfinder` class
2. Implement `apply_scenario()` — profile mutation for each scenario type
3. Implement beam search with pruning
4. Implement path scoring
5. Write `tests/test_career_pathfinder.py` with fixture-based tests

### Phase 2: API Integration (~1 day)
1. Add API models to `src/api/models.py`
2. Add `POST /api/career-path` endpoint to `src/api/app.py`
3. Add CLI command for testing
4. Integration tests

### Phase 3: Frontend (~2 days)
1. New "Career Path" component in Match Lab (or new tab)
2. Step-by-step visualization (vertical timeline with cards)
3. Each step shows: scenario title, salary change, jobs unlocked, evidence count
4. Cumulative metrics at the bottom
5. "Explore alternative paths" to compare top 3-5 strategies

## Verification Plan

### Unit Tests
- `apply_scenario()` correctly mutates profiles for all 5 scenario types
- Beam search explores correct number of paths
- Pruning rules trigger appropriately (diminishing returns, cost ceiling, cycles)
- Path scoring produces correct rankings
- Edge cases: empty scenarios, thin markets, single-step paths

### Integration Tests
- Full path generation with mock `CareerDeltaEngine`
- API endpoint returns well-formed responses
- CLI command exercises end-to-end flow

### Manual E2E Test
```bash
# Start API server
python -m src.cli api-serve

# Test via CLI
python -m src.cli career-path \
  --skills "Python,SQL,Excel" \
  --title "Data Analyst" \
  --target-salary 10000 \
  --depth 3

# Test via API
curl -X POST http://localhost:8000/api/career-path \
  -H "Content-Type: application/json" \
  -d '{
    "profile_text": "Data analyst with 3 years experience in Python and SQL",
    "current_title": "Data Analyst",
    "current_skills": ["Python", "SQL", "Excel"],
    "target_salary_min": 10000,
    "max_depth": 3
  }'
```

### Performance Test
- Verify total latency < 3 seconds for depth-3 paths
- Verify beam width 3 produces meaningfully different paths
- Verify pruning reduces evaluation count by >50%
