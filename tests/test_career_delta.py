from src.mcf.career_delta import (
    BaselineMarketPosition,
    CareerDeltaCandidate,
    CareerDeltaCandidatePool,
    CareerDeltaDependencies,
    CareerDeltaEngine,
    CareerDeltaRequest,
    ComputeBudget,
    MarketPosition,
    PivotScenarioSignal,
    SalaryBand,
    ScenarioChange,
    ScenarioConfidence,
    ScenarioSummary,
    ScenarioType,
    SkillScenarioSignal,
    build_filtered_scenario,
    build_scenario_id,
    rank_and_filter_scenarios,
)
from src.mcf.market_stats import MarketAggregate


class _TaxonomyStub:
    def normalize_title_family(self, title: str):
        return title


class _MarketStatsStub:
    def __init__(self, *, skills=None, title_families=None, industries=None, snapshot=None):
        self.skills = skills or {}
        self.title_families = title_families or {}
        self.industries = industries or {}
        self.snapshot = snapshot or {"current_industry": None, "current_title_family": None}

    def get_market_snapshot(self, request):
        return self.snapshot

    def get_skill_stats(self, skill: str):
        return self.skills.get(
            skill,
            MarketAggregate(key=skill.lower(), label=skill, kind="skill"),
        )

    def get_title_family_stats(self, title_or_family: str):
        return self.title_families.get(
            title_or_family,
            self.title_families.get(
                title_or_family.lower(),
                MarketAggregate(key=title_or_family.lower(), label=title_or_family, kind="title_family"),
            ),
        )

    def get_industry_stats(self, industry: str):
        return self.industries.get(
            industry,
            MarketAggregate(key=industry.lower(), label=industry, kind="industry"),
        )


class _SearchScoringStub:
    def __init__(self, *, degraded: bool = False, pool=None, related_skills=None):
        self.degraded = degraded
        self.pool = pool or CareerDeltaCandidatePool(degraded=degraded)
        self.related_skills = related_skills or {}

    def build_candidate_pool(self, request):
        return self.pool

    def get_related_skills(self, skill: str, k: int = 10):
        return tuple(self.related_skills.get(skill, ()))[:k]


def _candidate(
    *,
    uuid: str,
    title: str = "Data Engineer",
    title_family: str = "data-engineer",
    industry_key: str = "technology/software_and_platforms",
    industry_label: str = "technology / software_and_platforms",
    overall_fit: float = 0.72,
    skills: tuple[str, ...] = (),
    gap_skills: tuple[str, ...] = (),
):
    return CareerDeltaCandidate(
        uuid=uuid,
        title=title,
        company_name="Example Co",
        title_family=title_family,
        industry_key=industry_key,
        industry_label=industry_label,
        overall_fit=overall_fit,
        retrieval_score=0.8,
        matched_skills=("Python",),
        gap_skills=gap_skills,
        skills=skills,
    )


class TestScenarioIds:
    def test_scenario_id_is_stable_for_same_inputs(self):
        scenario_id_a = build_scenario_id(
            ScenarioType.SAME_ROLE,
            source_title_family="software-engineer",
            target_title_family="software-engineer",
            target_sector="technology",
            market_position=MarketPosition.COMPETITIVE,
        )
        scenario_id_b = build_scenario_id(
            ScenarioType.SAME_ROLE,
            source_title_family="software-engineer",
            target_title_family="software-engineer",
            target_sector="technology",
            market_position=MarketPosition.COMPETITIVE,
        )

        assert scenario_id_a == scenario_id_b

    def test_scenario_id_changes_when_semantics_change(self):
        same_role_id = build_scenario_id(
            ScenarioType.SAME_ROLE,
            source_title_family="software-engineer",
            target_title_family="software-engineer",
            target_sector="technology",
            market_position=MarketPosition.COMPETITIVE,
        )
        pivot_id = build_scenario_id(
            ScenarioType.INDUSTRY_PIVOT,
            source_title_family="software-engineer",
            target_title_family="data-scientist",
            target_sector="technology",
            market_position=MarketPosition.STRETCH,
        )

        assert same_role_id != pivot_id


class TestCareerDeltaEngine:
    def test_analyze_returns_normalized_internal_contract(self):
        engine = CareerDeltaEngine()

        response = engine.analyze(
            CareerDeltaRequest(
                profile_text="  Staff engineer with platform and Python experience.  ",
                target_titles=("Platform Engineer", "Platform Engineer", "Site Reliability Engineer"),
            )
        )

        assert response.request.profile_text == "Staff engineer with platform and Python experience."
        assert response.request.target_titles == ("Platform Engineer", "Site Reliability Engineer")
        assert response.summaries == ()
        assert response.filtered_scenarios == ()
        assert response.degraded is True
        assert response.thin_market is False

    def test_engine_is_not_degraded_when_dependencies_are_supplied(self):
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(),
            search_scoring=_SearchScoringStub(),
        )
        engine = CareerDeltaEngine(dependencies)

        response = engine.analyze(CareerDeltaRequest(profile_text="Product manager"))

        assert response.degraded is False
        assert response.baseline is not None
        assert response.thin_market is True

    def test_engine_generates_skill_addition_scenarios_from_reachable_gap_skills(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    skills=("Python", "SQL", "Kubernetes"),
                    gap_skills=("Kubernetes",),
                ),
                _candidate(
                    uuid="2",
                    title="Platform Engineer",
                    skills=("Python", "SQL", "Kubernetes", "Terraform"),
                    gap_skills=("Kubernetes", "Terraform"),
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=20,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Kubernetes": MarketAggregate(
                        key="kubernetes",
                        label="Kubernetes",
                        kind="skill",
                        job_count=40,
                        median_salary_annual=210000,
                        momentum=0.14,
                    ),
                    "Terraform": MarketAggregate(
                        key="terraform",
                        label="Terraform",
                        kind="skill",
                        job_count=18,
                        median_salary_annual=195000,
                        momentum=0.06,
                    ),
                }
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data engineer with Python and SQL experience.",
                current_skills=("Python", "SQL"),
            )
        )

        additions = [summary for summary in response.summaries if summary.scenario_type == ScenarioType.SKILL_ADDITION]
        assert additions
        assert additions[0].title == "Add Kubernetes"
        assert additions[0].change == ScenarioChange(added_skills=("Kubernetes",))
        assert additions[0].signals[0].supporting_jobs == 2
        assert additions[0].signals[0].market_momentum == 0.14

    def test_engine_skips_skill_additions_without_any_profile_skill_baseline(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    skills=("Python", "SQL", "Kubernetes"),
                    gap_skills=("Kubernetes",),
                ),
                _candidate(
                    uuid="2",
                    title="Platform Engineer",
                    skills=("Python", "SQL", "Terraform"),
                    gap_skills=("Terraform",),
                ),
            ),
            total_candidates=12,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Kubernetes": MarketAggregate(
                        key="kubernetes",
                        label="Kubernetes",
                        kind="skill",
                        job_count=40,
                        median_salary_annual=210000,
                        momentum=0.14,
                    ),
                    "Terraform": MarketAggregate(
                        key="terraform",
                        label="Terraform",
                        kind="skill",
                        job_count=30,
                        median_salary_annual=205000,
                        momentum=0.1,
                    ),
                }
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(CareerDeltaRequest(profile_text="Vague manager summary."))

        assert response.summaries == ()

    def test_engine_requires_more_than_one_supporting_job_for_skill_additions(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    skills=("Python", "SQL", "Kubernetes"),
                    gap_skills=("Kubernetes",),
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=1,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Kubernetes": MarketAggregate(
                        key="kubernetes",
                        label="Kubernetes",
                        kind="skill",
                        job_count=40,
                        median_salary_annual=210000,
                        momentum=0.14,
                    ),
                }
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data engineer with Python and SQL experience.",
                current_skills=("Python", "SQL"),
            )
        )

        assert all(summary.scenario_type != ScenarioType.SKILL_ADDITION for summary in response.summaries)

    def test_engine_generates_skill_substitution_only_for_adjacent_improving_skills(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(uuid="1", skills=("Power BI", "SQL")),
                _candidate(uuid="2", title="BI Analyst", skills=("Power BI", "Python")),
            ),
            extracted_skills=("Excel", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Excel": MarketAggregate(
                        key="excel",
                        label="Excel",
                        kind="skill",
                        job_count=12,
                        median_salary_annual=120000,
                        momentum=0.01,
                    ),
                    "Power BI": MarketAggregate(
                        key="power bi",
                        label="Power BI",
                        kind="skill",
                        job_count=30,
                        median_salary_annual=150000,
                        momentum=0.12,
                    ),
                }
            ),
            search_scoring=_SearchScoringStub(
                pool=pool,
                related_skills={
                    "Excel": (
                        {"skill": "Power BI", "similarity": 0.91, "same_cluster": True},
                        {"skill": "Word", "similarity": 0.55, "same_cluster": False},
                    )
                },
            ),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Business analyst with Excel and SQL reporting experience.",
                current_skills=("Excel", "SQL"),
            )
        )

        substitutions = [
            summary for summary in response.summaries if summary.scenario_type == ScenarioType.SKILL_SUBSTITUTION
        ]
        assert substitutions
        assert substitutions[0].change.replaced_skills[0].from_skill == "Excel"
        assert substitutions[0].change.replaced_skills[0].to_skill == "Power BI"
        assert substitutions[0].signals[0].same_cluster is True

    def test_engine_skips_substitution_without_material_market_improvement(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(uuid="1", skills=("Tableau", "SQL")),
                _candidate(uuid="2", title="BI Analyst", skills=("Tableau", "Python")),
            ),
            extracted_skills=("Excel", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Excel": MarketAggregate(
                        key="excel",
                        label="Excel",
                        kind="skill",
                        job_count=20,
                        median_salary_annual=140000,
                        momentum=0.05,
                    ),
                    "Tableau": MarketAggregate(
                        key="tableau",
                        label="Tableau",
                        kind="skill",
                        job_count=22,
                        median_salary_annual=141000,
                        momentum=0.07,
                    ),
                }
            ),
            search_scoring=_SearchScoringStub(
                pool=pool,
                related_skills={"Excel": ({"skill": "Tableau", "similarity": 0.88, "same_cluster": True},)},
            ),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Business analyst with Excel and SQL reporting experience.",
                current_skills=("Excel", "SQL"),
            )
        )

        assert all(summary.scenario_type != ScenarioType.SKILL_SUBSTITUTION for summary in response.summaries)

    def test_engine_generates_grounded_title_pivot_scenarios(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.71,
                ),
                _candidate(
                    uuid="2",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.69,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                title_families={
                    "data-analyst": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                    "data-scientist": MarketAggregate(
                        key="data-scientist",
                        label="data-scientist",
                        kind="title_family",
                        job_count=35,
                        median_salary_annual=145000,
                        momentum=0.11,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=50,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
            )
        )

        pivots = [summary for summary in response.summaries if summary.scenario_type == ScenarioType.TITLE_PIVOT]
        assert pivots
        assert pivots[0].target_title == "Data Scientist"
        assert pivots[0].change.target_title_family == "data-scientist"

    def test_engine_caps_title_pivots_to_the_single_best_family(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.72,
                ),
                _candidate(
                    uuid="2",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.7,
                ),
                _candidate(
                    uuid="3",
                    title="Product Manager",
                    title_family="product-manager",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.66,
                ),
                _candidate(
                    uuid="4",
                    title="Product Manager",
                    title_family="product-manager",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.64,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=24,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                title_families={
                    "data-analyst": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                    "data-scientist": MarketAggregate(
                        key="data-scientist",
                        label="data-scientist",
                        kind="title_family",
                        job_count=38,
                        median_salary_annual=148000,
                        momentum=0.12,
                    ),
                    "product-manager": MarketAggregate(
                        key="product-manager",
                        label="product-manager",
                        kind="title_family",
                        job_count=26,
                        median_salary_annual=138000,
                        momentum=0.07,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=50,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
                limit=6,
            )
        )

        pivots = [summary for summary in response.summaries if summary.scenario_type == ScenarioType.TITLE_PIVOT]

        assert len(pivots) == 1
        assert pivots[0].change.target_title_family == "data-scientist"

    def test_engine_dedupes_unknown_industry_title_pivots_by_title_family(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.71,
                ),
                _candidate(
                    uuid="2",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="financial_services/banking",
                    industry_label="financial_services / banking",
                    overall_fit=0.69,
                ),
                _candidate(
                    uuid="3",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.68,
                ),
                _candidate(
                    uuid="4",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="financial_services/banking",
                    industry_label="financial_services / banking",
                    overall_fit=0.67,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=20,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                title_families={
                    "data-analyst": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                    "data-scientist": MarketAggregate(
                        key="data-scientist",
                        label="data-scientist",
                        kind="title_family",
                        job_count=35,
                        median_salary_annual=145000,
                        momentum=0.11,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="unknown/unknown",
                        label="unknown",
                        kind="industry",
                        job_count=0,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
            )
        )

        pivots = [summary for summary in response.summaries if summary.scenario_type == ScenarioType.TITLE_PIVOT]
        assert len(pivots) == 1
        assert pivots[0].scenario_id.startswith("title_pivot:")

    def test_engine_generates_same_role_industry_pivots_as_first_class_scenarios(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Analyst",
                    title_family="data-analyst",
                    industry_key="financial_services/banking",
                    industry_label="financial_services / banking",
                    overall_fit=0.7,
                ),
                _candidate(
                    uuid="2",
                    title="Senior Data Analyst",
                    title_family="data-analyst",
                    industry_key="financial_services/banking",
                    industry_label="financial_services / banking",
                    overall_fit=0.68,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                industries={
                    "technology/data_and_ai": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.02,
                    ),
                    "financial_services/banking": MarketAggregate(
                        key="financial_services/banking",
                        label="financial_services / banking",
                        kind="industry",
                        job_count=40,
                        median_salary_annual=150000,
                        momentum=0.1,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.02,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
            )
        )

        pivots = [
            summary for summary in response.summaries if summary.scenario_type == ScenarioType.SAME_ROLE_INDUSTRY_PIVOT
        ]
        assert pivots
        assert pivots[0].change.source_industry == "technology/data_and_ai"
        assert pivots[0].change.target_industry == "financial_services/banking"

    def test_engine_skips_same_role_industry_pivot_on_trivial_job_count_delta(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Analyst",
                    title_family="data-analyst",
                    industry_key="technology/software_and_platforms",
                    industry_label="technology / software_and_platforms",
                    overall_fit=0.7,
                ),
                _candidate(
                    uuid="2",
                    title="Senior Data Analyst",
                    title_family="data-analyst",
                    industry_key="technology/software_and_platforms",
                    industry_label="technology / software_and_platforms",
                    overall_fit=0.69,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                industries={
                    "technology/data_and_ai": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.05,
                    ),
                    "technology/software_and_platforms": MarketAggregate(
                        key="technology/software_and_platforms",
                        label="technology / software_and_platforms",
                        kind="industry",
                        job_count=21,
                        median_salary_annual=120000,
                        momentum=0.05,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=20,
                        median_salary_annual=120000,
                        momentum=0.05,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
            )
        )

        assert all(summary.scenario_type != ScenarioType.SAME_ROLE_INDUSTRY_PIVOT for summary in response.summaries)

    def test_engine_generates_adjacent_role_industry_pivots_within_distance_limit(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="financial_services/insurance",
                    industry_label="financial_services / insurance",
                    overall_fit=0.69,
                ),
                _candidate(
                    uuid="2",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="financial_services/insurance",
                    industry_label="financial_services / insurance",
                    overall_fit=0.67,
                ),
            ),
            extracted_skills=("Excel", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                industries={
                    "financial_services/banking": MarketAggregate(
                        key="financial_services/banking",
                        label="financial_services / banking",
                        kind="industry",
                        job_count=22,
                        median_salary_annual=115000,
                        momentum=0.02,
                    ),
                    "financial_services/insurance": MarketAggregate(
                        key="financial_services/insurance",
                        label="financial_services / insurance",
                        kind="industry",
                        job_count=35,
                        median_salary_annual=135000,
                        momentum=0.09,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="financial_services/banking",
                        label="financial_services / banking",
                        kind="industry",
                        job_count=22,
                        median_salary_annual=115000,
                        momentum=0.02,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=15,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Excel and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Excel", "SQL"),
            )
        )

        pivots = [
            summary
            for summary in response.summaries
            if summary.scenario_type == ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT
        ]
        assert pivots
        assert pivots[0].target_title == "Data Scientist"
        assert pivots[0].change.target_industry == "financial_services/insurance"

    def test_engine_filters_adjacent_role_industry_pivots_that_are_too_far(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(
                    uuid="1",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="healthcare/clinical_and_care_delivery",
                    industry_label="healthcare / clinical_and_care_delivery",
                    overall_fit=0.69,
                ),
                _candidate(
                    uuid="2",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="healthcare/clinical_and_care_delivery",
                    industry_label="healthcare / clinical_and_care_delivery",
                    overall_fit=0.68,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=18,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                industries={
                    "technology/data_and_ai": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=25,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                    "healthcare/clinical_and_care_delivery": MarketAggregate(
                        key="healthcare/clinical_and_care_delivery",
                        label="healthcare / clinical_and_care_delivery",
                        kind="industry",
                        job_count=30,
                        median_salary_annual=150000,
                        momentum=0.11,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        key="technology/data_and_ai",
                        label="technology / data_and_ai",
                        kind="industry",
                        job_count=25,
                        median_salary_annual=120000,
                        momentum=0.03,
                    ),
                    "current_title_family": MarketAggregate(
                        key="data-analyst",
                        label="data-analyst",
                        kind="title_family",
                        job_count=20,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
            )
        )

        assert response.filtered_scenarios
        assert response.filtered_scenarios[0].scenario_type == ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT
        assert response.filtered_scenarios[0].reason_code == "industry_distance_too_high"

    def test_filtered_scenario_uses_stable_id_helper(self):
        filtered = build_filtered_scenario(
            scenario_type=ScenarioType.INDUSTRY_PIVOT,
            reason_code="low_evidence",
            explanation="Not enough market support for a confident pivot.",
            confidence=ScenarioConfidence(score=0.32, evidence_coverage=0.2, market_sample_size=4),
            source_title_family="software-engineer",
            target_title_family="product-manager",
            target_sector="commercial",
            market_position=MarketPosition.UNCLEAR,
        )

        assert filtered.scenario_id.startswith("industry_pivot:")
        assert filtered.reason_code == "low_evidence"


class TestScenarioRanking:
    def test_ranker_prefers_grounded_skill_addition_over_higher_upside_title_pivot(self):
        baseline = BaselineMarketPosition(
            position=MarketPosition.COMPETITIVE,
            reachable_jobs=12,
            total_candidates=20,
            fit_median=0.55,
            fit_p90=0.75,
            salary_band=SalaryBand(median_annual=120000),
        )
        skill_summary = ScenarioSummary(
            scenario_id="skill_addition:kubernetes",
            scenario_type=ScenarioType.SKILL_ADDITION,
            title="Add Kubernetes",
            summary="Add Kubernetes to strengthen reachable platform roles.",
            market_position=MarketPosition.COMPETITIVE,
            confidence=ScenarioConfidence(score=0.82, evidence_coverage=0.6, market_sample_size=30),
            change=ScenarioChange(added_skills=("Kubernetes",)),
            signals=(
                SkillScenarioSignal(
                    skill="Kubernetes",
                    supporting_jobs=5,
                    supporting_share_pct=40.0,
                    market_job_count=30,
                    market_momentum=0.1,
                    salary_lift_pct=0.08,
                ),
            ),
            expected_salary_delta_pct=0.08,
        )
        title_pivot = ScenarioSummary(
            scenario_id="title_pivot:platform",
            scenario_type=ScenarioType.TITLE_PIVOT,
            title="Pivot toward Platform Engineer",
            summary="Use adjacent-role demand to pivot toward platform engineering.",
            market_position=MarketPosition.STRETCH,
            confidence=ScenarioConfidence(score=0.78, evidence_coverage=0.5, market_sample_size=24),
            change=ScenarioChange(
                source_title_family="data-engineer",
                target_title_family="platform-engineer",
            ),
            signals=(
                PivotScenarioSignal(
                    supporting_jobs=4,
                    supporting_share_pct=30.0,
                    target_title_family="platform-engineer",
                    target_industry="technology/software_and_platforms",
                    title_distance="adjacent",
                    industry_distance=0,
                    fit_median=0.68,
                    market_job_count=24,
                    market_momentum=0.14,
                    salary_lift_pct=0.12,
                ),
            ),
            expected_salary_delta_pct=0.12,
        )

        ranked, _, _ = rank_and_filter_scenarios(
            (skill_summary, title_pivot),
            baseline=baseline,
            request=CareerDeltaRequest(profile_text="Profile", limit=5),
            budget=ComputeBudget(),
            started_at=0.0,
            clock=lambda: 0.0,
        )

        assert [summary.scenario_id for summary in ranked] == [
            "skill_addition:kubernetes",
            "title_pivot:platform",
        ]

    def test_ranker_prunes_exact_duplicates_predictably(self):
        duplicate_a = ScenarioSummary(
            scenario_id="skill_addition:a",
            scenario_type=ScenarioType.SKILL_ADDITION,
            title="Add Kubernetes",
            summary="Add Kubernetes",
            market_position=MarketPosition.COMPETITIVE,
            confidence=ScenarioConfidence(score=0.8, evidence_coverage=0.6, market_sample_size=20),
            change=ScenarioChange(added_skills=("Kubernetes",)),
            signals=(
                SkillScenarioSignal(
                    skill="Kubernetes",
                    supporting_jobs=4,
                    supporting_share_pct=50.0,
                    market_job_count=30,
                    market_momentum=0.1,
                    salary_lift_pct=0.1,
                ),
            ),
            expected_salary_delta_pct=0.1,
        )
        duplicate_b = ScenarioSummary(
            scenario_id="skill_addition:b",
            scenario_type=ScenarioType.SKILL_ADDITION,
            title="Add Kubernetes",
            summary="Add Kubernetes",
            market_position=MarketPosition.COMPETITIVE,
            confidence=ScenarioConfidence(score=0.6, evidence_coverage=0.5, market_sample_size=18),
            change=ScenarioChange(added_skills=("Kubernetes",)),
            signals=(
                SkillScenarioSignal(
                    skill="Kubernetes",
                    supporting_jobs=3,
                    supporting_share_pct=40.0,
                    market_job_count=28,
                    market_momentum=0.08,
                    salary_lift_pct=0.08,
                ),
            ),
            expected_salary_delta_pct=0.08,
        )

        ranked, filtered, degraded = rank_and_filter_scenarios(
            (duplicate_a, duplicate_b),
            baseline=BaselineMarketPosition(
                position=MarketPosition.COMPETITIVE,
                reachable_jobs=12,
                total_candidates=20,
                fit_median=0.6,
                fit_p90=0.75,
                salary_band=SalaryBand(median_annual=120000),
            ),
            request=CareerDeltaRequest(profile_text="Profile", limit=5),
            budget=ComputeBudget(),
            started_at=0.0,
            clock=lambda: 0.0,
        )

        assert degraded is False
        assert len(ranked) == 1
        assert ranked[0].scenario_id == "skill_addition:a"
        assert filtered[0].reason_code == "duplicate_scenario"

    def test_ranker_prunes_cross_type_semantic_duplicates(self):
        baseline = BaselineMarketPosition(
            position=MarketPosition.COMPETITIVE,
            reachable_jobs=12,
            total_candidates=20,
            fit_median=0.6,
            fit_p90=0.75,
            salary_band=SalaryBand(median_annual=120000),
        )
        shared_change = ScenarioChange(
            added_skills=("Kubernetes",),
            source_title_family="data-engineer",
            target_title_family="platform-engineer",
        )
        skill_summary = ScenarioSummary(
            scenario_id="skill_addition:a",
            scenario_type=ScenarioType.SKILL_ADDITION,
            title="Add Kubernetes",
            summary="Add Kubernetes to move toward platform engineering.",
            market_position=MarketPosition.COMPETITIVE,
            confidence=ScenarioConfidence(score=0.82, evidence_coverage=0.62, market_sample_size=24),
            change=shared_change,
            signals=(
                SkillScenarioSignal(
                    skill="Kubernetes",
                    supporting_jobs=4,
                    supporting_share_pct=40.0,
                    market_job_count=32,
                    market_momentum=0.14,
                    salary_lift_pct=0.1,
                ),
            ),
            expected_salary_delta_pct=0.1,
        )
        pivot_summary = ScenarioSummary(
            scenario_id="title_pivot:a",
            scenario_type=ScenarioType.TITLE_PIVOT,
            title="Pivot toward Platform Engineer",
            summary="Use Kubernetes strength to pivot toward platform engineering.",
            market_position=MarketPosition.STRETCH,
            confidence=ScenarioConfidence(score=0.74, evidence_coverage=0.48, market_sample_size=22),
            change=shared_change,
            signals=(
                PivotScenarioSignal(
                    supporting_jobs=4,
                    supporting_share_pct=40.0,
                    target_title_family="platform-engineer",
                    target_industry="technology/platform",
                    title_distance="adjacent",
                    industry_distance=0,
                    fit_median=0.66,
                    market_job_count=24,
                    market_momentum=0.11,
                    salary_lift_pct=0.08,
                ),
            ),
            expected_salary_delta_pct=0.08,
        )

        ranked, filtered, degraded = rank_and_filter_scenarios(
            (skill_summary, pivot_summary),
            baseline=baseline,
            request=CareerDeltaRequest(profile_text="Profile", limit=5),
            budget=ComputeBudget(),
            started_at=0.0,
            clock=lambda: 0.0,
        )

        assert degraded is False
        assert len(ranked) == 1
        assert ranked[0].scenario_id == "skill_addition:a"
        assert len(filtered) == 1
        assert filtered[0].scenario_type is ScenarioType.TITLE_PIVOT
        assert filtered[0].reason_code == "overlapping_scenario"

    def test_ranker_enforces_type_diversity_before_fill(self):
        baseline = BaselineMarketPosition(
            position=MarketPosition.COMPETITIVE,
            reachable_jobs=12,
            total_candidates=20,
            fit_median=0.55,
            fit_p90=0.75,
            salary_band=SalaryBand(median_annual=120000),
        )
        summaries = (
            ScenarioSummary(
                scenario_id="skill:1",
                scenario_type=ScenarioType.SKILL_ADDITION,
                title="Add Kubernetes",
                summary="Add Kubernetes",
                market_position=MarketPosition.COMPETITIVE,
                confidence=ScenarioConfidence(score=0.9, evidence_coverage=0.6, market_sample_size=20),
                change=ScenarioChange(added_skills=("Kubernetes",)),
                signals=(SkillScenarioSignal("Kubernetes", 5, 50.0, 30, market_momentum=0.15, salary_lift_pct=0.12),),
                expected_salary_delta_pct=0.12,
            ),
            ScenarioSummary(
                scenario_id="skill:2",
                scenario_type=ScenarioType.SKILL_ADDITION,
                title="Add Terraform",
                summary="Add Terraform",
                market_position=MarketPosition.COMPETITIVE,
                confidence=ScenarioConfidence(score=0.88, evidence_coverage=0.55, market_sample_size=18),
                change=ScenarioChange(added_skills=("Terraform",)),
                signals=(SkillScenarioSignal("Terraform", 4, 40.0, 26, market_momentum=0.12, salary_lift_pct=0.1),),
                expected_salary_delta_pct=0.1,
            ),
            ScenarioSummary(
                scenario_id="pivot:1",
                scenario_type=ScenarioType.TITLE_PIVOT,
                title="Pivot toward Data Scientist",
                summary="Pivot",
                market_position=MarketPosition.STRETCH,
                confidence=ScenarioConfidence(score=0.7, evidence_coverage=0.4, market_sample_size=22),
                change=ScenarioChange(source_title_family="data-analyst", target_title_family="data-scientist"),
                signals=(
                    PivotScenarioSignal(
                        supporting_jobs=4,
                        supporting_share_pct=40.0,
                        target_title_family="data-scientist",
                        target_industry="technology/data_and_ai",
                        title_distance="adjacent",
                        industry_distance=0,
                        fit_median=0.68,
                        market_job_count=22,
                        market_momentum=0.09,
                        salary_lift_pct=0.08,
                    ),
                ),
                expected_salary_delta_pct=0.08,
            ),
        )

        ranked, _, _ = rank_and_filter_scenarios(
            summaries,
            baseline=baseline,
            request=CareerDeltaRequest(profile_text="Profile", limit=2),
            budget=ComputeBudget(),
            started_at=0.0,
            clock=lambda: 0.0,
        )

        assert len(ranked) == 2
        assert {item.scenario_type for item in ranked} == {
            ScenarioType.SKILL_ADDITION,
            ScenarioType.TITLE_PIVOT,
        }

    def test_ranker_prefers_same_role_industry_pivot_over_adjacent_role_when_upside_is_similar(self):
        baseline = BaselineMarketPosition(
            position=MarketPosition.COMPETITIVE,
            reachable_jobs=12,
            total_candidates=20,
            fit_median=0.55,
            fit_p90=0.75,
            salary_band=SalaryBand(median_annual=120000),
        )
        same_role = ScenarioSummary(
            scenario_id="same_role:industry",
            scenario_type=ScenarioType.SAME_ROLE_INDUSTRY_PIVOT,
            title="Keep the role, pivot into technology / data_and_ai",
            summary="Stay in the same role family inside a stronger sector.",
            market_position=MarketPosition.COMPETITIVE,
            confidence=ScenarioConfidence(score=0.79, evidence_coverage=0.5, market_sample_size=28),
            change=ScenarioChange(
                source_title_family="data-scientist",
                target_title_family="data-scientist",
                source_industry="financial_services/banking",
                target_industry="technology/data_and_ai",
            ),
            signals=(
                PivotScenarioSignal(
                    supporting_jobs=4,
                    supporting_share_pct=30.0,
                    target_title_family="data-scientist",
                    target_industry="technology/data_and_ai",
                    title_distance="same",
                    industry_distance=1,
                    fit_median=0.67,
                    market_job_count=28,
                    market_momentum=0.12,
                    salary_lift_pct=0.1,
                ),
            ),
            expected_salary_delta_pct=0.1,
        )
        adjacent_role = ScenarioSummary(
            scenario_id="adjacent_role:industry",
            scenario_type=ScenarioType.ADJACENT_ROLE_INDUSTRY_PIVOT,
            title="Move toward ML Engineer in technology / data_and_ai",
            summary="Take an adjacent role and sector move with similar upside.",
            market_position=MarketPosition.STRETCH,
            confidence=ScenarioConfidence(score=0.77, evidence_coverage=0.5, market_sample_size=30),
            change=ScenarioChange(
                source_title_family="data-scientist",
                target_title_family="ml-engineer",
                source_industry="financial_services/banking",
                target_industry="technology/data_and_ai",
            ),
            signals=(
                PivotScenarioSignal(
                    supporting_jobs=4,
                    supporting_share_pct=30.0,
                    target_title_family="ml-engineer",
                    target_industry="technology/data_and_ai",
                    title_distance="adjacent",
                    industry_distance=1,
                    fit_median=0.68,
                    market_job_count=30,
                    market_momentum=0.13,
                    salary_lift_pct=0.11,
                ),
            ),
            expected_salary_delta_pct=0.11,
        )

        ranked, _, _ = rank_and_filter_scenarios(
            (same_role, adjacent_role),
            baseline=baseline,
            request=CareerDeltaRequest(profile_text="Profile", limit=5),
            budget=ComputeBudget(),
            started_at=0.0,
            clock=lambda: 0.0,
        )

        assert [summary.scenario_id for summary in ranked] == [
            "same_role:industry",
            "adjacent_role:industry",
        ]

    def test_engine_marks_response_degraded_when_budget_truncates_scoring(self):
        pool = CareerDeltaCandidatePool(
            candidates=(
                _candidate(uuid="1", skills=("Python", "SQL", "Kubernetes"), gap_skills=("Kubernetes",)),
                _candidate(
                    uuid="2",
                    skills=("Python", "SQL", "Kubernetes", "Terraform"),
                    gap_skills=("Kubernetes", "Terraform"),
                ),
                _candidate(
                    uuid="3",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.69,
                ),
                _candidate(
                    uuid="4",
                    title="Data Scientist",
                    title_family="data-scientist",
                    industry_key="technology/data_and_ai",
                    industry_label="technology / data_and_ai",
                    overall_fit=0.67,
                ),
            ),
            extracted_skills=("Python", "SQL"),
            total_candidates=20,
        )
        dependencies = CareerDeltaDependencies(
            taxonomy=_TaxonomyStub(),
            market_stats=_MarketStatsStub(
                skills={
                    "Kubernetes": MarketAggregate(
                        "kubernetes",
                        "Kubernetes",
                        "skill",
                        40,
                        210000,
                        0.14,
                    ),
                    "Terraform": MarketAggregate(
                        "terraform",
                        "Terraform",
                        "skill",
                        30,
                        205000,
                        0.1,
                    ),
                },
                title_families={
                    "data-analyst": MarketAggregate(
                        "data-analyst",
                        "data-analyst",
                        "title_family",
                        20,
                        120000,
                        0.03,
                    ),
                    "data-scientist": MarketAggregate(
                        "data-scientist",
                        "data-scientist",
                        "title_family",
                        35,
                        145000,
                        0.11,
                    ),
                },
                snapshot={
                    "current_industry": MarketAggregate(
                        "technology/data_and_ai",
                        "technology / data_and_ai",
                        "industry",
                        50,
                    ),
                    "current_title_family": MarketAggregate(
                        "data-analyst",
                        "data-analyst",
                        "title_family",
                        20,
                        120000,
                        0.03,
                    ),
                },
            ),
            search_scoring=_SearchScoringStub(pool=pool),
        )

        response = CareerDeltaEngine(
            dependencies,
            budget=ComputeBudget(max_wall_time_ms=5000, max_scenarios_evaluated=1),
        ).analyze(
            CareerDeltaRequest(
                profile_text="Data analyst with Python and SQL experience.",
                current_title="Data Analyst",
                current_skills=("Python", "SQL"),
                limit=3,
            )
        )

        assert response.degraded is True
        assert response.filtered_scenarios
        assert any(item.reason_code == "budget_exhausted" for item in response.filtered_scenarios)
