from src.mcf.career_delta import (
    CareerDeltaDependencies,
    CareerDeltaEngine,
    CareerDeltaRequest,
    MarketPosition,
    ScenarioConfidence,
    ScenarioType,
    build_filtered_scenario,
    build_scenario_id,
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
            taxonomy=object(),
            market_stats=object(),
            search_scoring=object(),
        )
        engine = CareerDeltaEngine(dependencies)

        response = engine.analyze(CareerDeltaRequest(profile_text="Product manager"))

        assert response.degraded is False
        assert response.thin_market is False

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
