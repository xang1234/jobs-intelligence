"""
Wide-pool retrieval provider for the career-delta baseline pass.

This module reuses the semantic search engine's existing profile-match helpers
to keep fit calculations aligned with Match Lab semantics while retaining
enough per-job evidence for later scenario rescoring.
"""

from __future__ import annotations

from collections import defaultdict

from .career_delta import CareerDeltaCandidate, CareerDeltaCandidatePool, CareerDeltaRequest
from .embeddings.models import SearchRequest
from .embeddings.search_engine import SemanticSearchEngine
from .industry_taxonomy import (
    IndustryClassification,
    classification_from_bucket,
    classify_industry,
    normalize_categories,
    normalize_title_family,
)


class SearchEngineCareerDeltaProvider:
    """Build one reusable candidate pool from a broad profile retrieval pass."""

    def __init__(
        self,
        engine: SemanticSearchEngine,
        *,
        retrieval_multiplier: int = 8,
        minimum_pool_size: int = 120,
    ):
        self.engine = engine
        self.retrieval_multiplier = retrieval_multiplier
        self.minimum_pool_size = minimum_pool_size

    def build_candidate_pool(self, request: CareerDeltaRequest) -> CareerDeltaCandidatePool:
        if not self.engine._loaded:
            self.engine.load()

        extracted_skills = tuple(self.engine._extract_skills_from_text(request.profile_text))
        normalized_titles = [title.lower() for title in request.normalized_target_titles() if title.strip()]
        profile_level = self.engine._infer_profile_seniority(request.profile_text, list(request.target_titles))
        search_limit = max(request.limit * self.retrieval_multiplier, self.minimum_pool_size)

        if self.engine._has_vector_index and not self.engine._degraded:
            profile_embedding = self.engine._get_query_embedding(request.profile_text)
            search_k = max(search_limit * 2, 300)
            candidate_rows = self.engine.vector_backend.search_jobs(profile_embedding, k=search_k)
        else:
            keyword_seed = " ".join(extracted_skills[:5]) or request.profile_text[:120]
            fallback = self.engine._keyword_fallback_search(
                SearchRequest(
                    query=keyword_seed,
                    region=request.location,
                    limit=max(search_limit, 100),
                ),
                0.0,
            )
            candidate_rows = [(job.uuid, job.similarity_score) for job in fallback.results]

        jobs = self.engine.db.get_jobs_bulk([uuid for uuid, _ in candidate_rows])
        company_direct_industries = self._company_direct_industries(jobs.values())
        candidates: list[CareerDeltaCandidate] = []

        for uuid, raw_retrieval in candidate_rows:
            job = jobs.get(uuid)
            if not job:
                continue
            if request.location and job.get("region") != request.location:
                continue

            title_match = bool(
                normalized_titles and any(title in (job.get("title", "").lower()) for title in normalized_titles)
            )

            job_skills = tuple(self.engine._parse_skills(job.get("skills")))
            matched_skills = tuple(sorted(set(extracted_skills).intersection(job_skills)))
            missing_skills = tuple(sorted(set(extracted_skills) - set(job_skills)))[:10]
            gap_skills = tuple(sorted(set(job_skills) - set(extracted_skills)))[:10]
            skill_overlap = len(matched_skills) / len(extracted_skills) if extracted_skills else 0.0

            semantic_score = (
                float(raw_retrieval) if self.engine._has_vector_index and not self.engine._degraded else 0.0
            )
            keyword_score = float(raw_retrieval) if self.engine._degraded else 0.0
            seniority_fit = self.engine._score_seniority(
                profile_level,
                self.engine._normalize_seniority(job.get("seniority")),
            )
            salary_fit = self.engine._score_salary_alignment(
                request.target_salary_min,
                job.get("salary_annual_min"),
                job.get("salary_annual_max"),
            )

            weighted_parts = [
                (0.45, semantic_score if not self.engine._degraded else keyword_score),
                (0.35, skill_overlap),
                (0.10, seniority_fit),
            ]
            if request.target_salary_min is not None and salary_fit is not None:
                weighted_parts.append((0.10, salary_fit))

            total_weight = sum(weight for weight, _ in weighted_parts) or 1.0
            overall_fit = sum(weight * value for weight, value in weighted_parts) / total_weight

            company_name = (job.get("company_name") or "").strip()
            industry = classification_from_bucket(job.get("industry_bucket"))
            if industry.is_unknown:
                industry = classify_industry(
                    normalize_categories(self._split_csv(job.get("categories"))),
                    company_classifications=company_direct_industries.get(company_name, ()),
                    skills=job_skills,
                )
            title_family = (job.get("title_family") or "").strip() or normalize_title_family(
                job.get("title", "")
            ).canonical

            candidates.append(
                CareerDeltaCandidate(
                    uuid=job["uuid"],
                    title=job.get("title", ""),
                    company_name=company_name,
                    title_family=title_family,
                    target_title_match=title_match,
                    industry_key=f"{industry.sector}/{industry.subsector}",
                    industry_label=f"{industry.sector} / {industry.subsector}",
                    overall_fit=round(float(overall_fit), 4),
                    retrieval_score=round(float(raw_retrieval), 4),
                    semantic_score=round(float(semantic_score), 4) if not self.engine._degraded else None,
                    bm25_score=round(float(keyword_score), 4) if self.engine._degraded else None,
                    skill_overlap_score=round(float(skill_overlap), 4),
                    seniority_fit=round(float(seniority_fit), 4),
                    salary_fit=round(float(salary_fit), 4) if salary_fit is not None else None,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    gap_skills=gap_skills,
                    skills=job_skills,
                    categories=tuple(self._split_csv(job.get("categories"))),
                    salary_annual_min=job.get("salary_annual_min"),
                    salary_annual_max=job.get("salary_annual_max"),
                    employment_type=job.get("employment_type"),
                    seniority=job.get("seniority"),
                    region=job.get("region"),
                    location=job.get("location"),
                    posted_date=job.get("posted_date"),
                    job_url=job.get("job_url"),
                )
            )

        candidates.sort(key=lambda candidate: candidate.overall_fit, reverse=True)
        limited = tuple(candidates[:search_limit])
        return CareerDeltaCandidatePool(
            candidates=limited,
            extracted_skills=extracted_skills,
            total_candidates=len(candidates),
            degraded=self.engine._degraded,
        )

    def get_related_skills(self, skill: str, k: int = 10):
        """Return related-skill metadata with a case-insensitive lookup fallback."""
        related = self.engine.get_related_skills(skill, k=k)
        if related is not None:
            return tuple(related.get("related", ()))

        normalized = self._canonical_skill(skill)
        if normalized != skill:
            related = self.engine.get_related_skills(normalized, k=k)
            if related is not None:
                return tuple(related.get("related", ()))
        return ()

    @staticmethod
    def _split_csv(raw_value) -> list[str]:
        if not raw_value:
            return []
        return [item.strip() for item in str(raw_value).split(",") if item.strip()]

    @staticmethod
    def _company_direct_industries(job_rows) -> dict[str, list[IndustryClassification]]:
        by_company: dict[str, list[IndustryClassification]] = defaultdict(list)
        for row in job_rows:
            company_name = (row.get("company_name") or "").strip()
            if not company_name:
                continue
            direct = classification_from_bucket(row.get("industry_bucket"))
            if direct.is_unknown:
                direct = classify_industry(SearchEngineCareerDeltaProvider._split_csv(row.get("categories")))
            if not direct.is_unknown:
                by_company[company_name].append(direct)
        return by_company

    def _canonical_skill(self, skill: str) -> str:
        query_expander = getattr(self.engine, "query_expander", None)
        if query_expander and hasattr(query_expander, "_skill_lower_map"):
            return query_expander._skill_lower_map.get(skill.lower(), skill)
        vector_manager = getattr(self.engine.vector_backend, "manager", None)
        skill_to_idx = getattr(vector_manager, "skill_to_idx", {})
        for known_skill in skill_to_idx.keys():
            if known_skill.lower() == skill.lower():
                return known_skill
        return skill
