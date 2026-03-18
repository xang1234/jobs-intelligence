from src.mcf.career_delta import (
    CareerDeltaCandidate,
    CareerDeltaCandidatePool,
    CareerDeltaDependencies,
    CareerDeltaEngine,
    CareerDeltaRequest,
    MarketPosition,
    ScenarioChange,
    ScenarioConfidence,
    ScenarioType,
    build_filtered_scenario,
    build_scenario_id,
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

        response = CareerDeltaEngine(dependencies).analyze(
            CareerDeltaRequest(profile_text="Vague manager summary.")
        )

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
            summary
            for summary in response.summaries
            if summary.scenario_type == ScenarioType.SAME_ROLE_INDUSTRY_PIVOT
        ]
        assert pivots
        assert pivots[0].change.source_industry == "technology/data_and_ai"
        assert pivots[0].change.target_industry == "financial_services/banking"

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
