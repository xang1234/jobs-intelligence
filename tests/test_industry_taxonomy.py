from src.mcf.industry_taxonomy import (
    IndustryClassification,
    IndustrySource,
    classify_industry,
    industry_distance,
    infer_company_dominant_industry,
    is_adjacent_role,
    is_same_role,
    normalize_categories,
    normalize_title_family,
)


class TestCategoryNormalization:
    def test_normalize_single_and_multi_category_strings(self):
        assert normalize_categories(["Information Technology / Data Science"]) == [
            "information technology",
            "data science",
        ]

    def test_compound_category_with_and_stays_intact(self):
        assert normalize_categories(["Research and Development"]) == ["research and development"]

    def test_unknown_categories_are_preserved_but_not_forced(self):
        classification = classify_industry(["Space Policy"])

        assert classification.is_unknown is True
        assert classification.normalized_categories == ("space policy",)


class TestIndustryClassification:
    def test_direct_category_mapping_wins(self):
        classification = classify_industry(["Information Technology", "Engineering"])

        assert classification.sector == "technology"
        assert classification.subsector == "software_and_platforms"
        assert classification.source == IndustrySource.DIRECT_CATEGORY

    def test_company_dominant_industry_is_used_as_fallback(self):
        company_history = [
            IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("financial_services", "banking", IndustrySource.DIRECT_CATEGORY, 0.9),
        ]

        classification = classify_industry(
            ["Ambiguous Category"],
            company_classifications=company_history,
        )

        assert classification.sector == "technology"
        assert classification.subsector == "data_and_ai"
        assert classification.source == IndustrySource.COMPANY_DOMINANT

    def test_skill_affinity_requires_enough_evidence(self):
        classification = classify_industry(
            ["Unknown"],
            skills=["Python", "Machine Learning", "TensorFlow"],
        )

        assert classification.sector == "technology"
        assert classification.subsector == "data_and_ai"
        assert classification.source == IndustrySource.SKILL_AFFINITY

    def test_company_dominant_threshold_rejects_weak_majority(self):
        company_history = [
            IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("financial_services", "banking", IndustrySource.DIRECT_CATEGORY, 0.9),
            IndustryClassification("financial_services", "banking", IndustrySource.DIRECT_CATEGORY, 0.9),
        ]

        assert infer_company_dominant_industry(company_history) is None


class TestTitleNormalization:
    def test_title_family_strips_seniority_noise(self):
        family = normalize_title_family("Senior Software Engineer II")

        assert family.canonical == "software-engineer"

    def test_manager_titles_keep_role_anchor(self):
        family = normalize_title_family("Lead Product Manager")

        assert family.canonical == "product-manager"

    def test_obvious_variants_group_together(self):
        assert is_same_role("Backend Developer", "Back-End Engineer") is True

    def test_adjacent_role_distinguishes_close_neighbors(self):
        assert is_adjacent_role("Data Scientist", "Data Analyst") is True
        assert is_adjacent_role("Data Scientist", "Nurse Educator") is False


class TestIndustryDistance:
    def test_distance_semantics(self):
        left = IndustryClassification("technology", "data_and_ai", IndustrySource.DIRECT_CATEGORY, 0.9)
        same_sector = IndustryClassification(
            "technology",
            "software_and_platforms",
            IndustrySource.DIRECT_CATEGORY,
            0.9,
        )
        different_sector = IndustryClassification(
            "healthcare",
            "clinical_and_care_delivery",
            IndustrySource.DIRECT_CATEGORY,
            0.9,
        )
        unknown = IndustryClassification()

        assert industry_distance(left, left) == 0
        assert industry_distance(left, same_sector) == 1
        assert industry_distance(left, different_sector) == 2
        assert industry_distance(left, unknown) == 3
