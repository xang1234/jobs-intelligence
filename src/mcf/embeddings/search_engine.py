"""
Semantic Search Engine for MCF job data.

Orchestrates hybrid semantic + keyword search by combining:
- SQL filtering (salary, location, employment type)
- Vector search (FAISS locally, pgvector on Postgres)
- FTS5 BM25 search (keyword relevance)
- Query expansion (synonym matching via skill clusters)
- Result caching (performance optimization)
- Graceful degradation (reliability when indexes unavailable)

Example:
    engine = SemanticSearchEngine(db_path="data/mcf_jobs.db")
    engine.load()

    response = engine.search(SearchRequest(
        query="machine learning engineer",
        salary_min=10000,
        limit=20
    ))

    for job in response.results:
        print(f"{job.title} at {job.company_name}: {job.similarity_score:.3f}")
"""

import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from cachetools import TTLCache

from ..db_factory import open_database
from .backends import DEFAULT_EMBEDDING_BACKEND, resolve_model_version
from .faiss_backend import FAISSVectorBackend
from .generator import EmbeddingGenerator
from .index_manager import (
    IndexCompatibilityError,
    IndexNotBuiltError,
)
from .models import (
    CompanySimilarity,
    CompanySimilarityRequest,
    JobResult,
    SearchExplanation,
    SearchRequest,
    SearchResponse,
    SimilarJobsRequest,
    SkillSearchRequest,
)
from .pgvector_backend import PGVectorBackend
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Orchestrates hybrid semantic + keyword search.

    Combines multiple search strategies:
    - SQL filtering for hard constraints (salary, location, type)
    - Vector search for semantic similarity
    - FTS5 BM25 search for keyword relevance
    - Query expansion for synonym matching
    - Result caching for performance
    - Graceful degradation for reliability

    The hybrid scoring formula is:
        score = alpha * semantic_score + (1 - alpha) * bm25_score + freshness_boost

    Example:
        engine = SemanticSearchEngine("data/mcf_jobs.db")
        engine.load()

        response = engine.search(SearchRequest(
            query="machine learning engineer",
            salary_min=10000,
            limit=20
        ))
    """

    # Default cache configuration
    QUERY_CACHE_SIZE = 1000
    QUERY_CACHE_TTL = 3600  # 1 hour
    RESULT_CACHE_SIZE = 200
    RESULT_CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        index_dir: Path = Path("data/embeddings"),
        model_version: str = "all-MiniLM-L6-v2",
        embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        onnx_model_dir: str | Path | None = None,
        search_backend: str | None = None,
        lean_hosted: bool = False,
    ):
        """
        Initialize the search engine.

        Args:
            db_path: Path or DSN for the configured database
            index_dir: Directory containing FAISS indexes
            model_version: Embedding model version for compatibility
            embedding_backend: Embedding inference backend for query encoding
            onnx_model_dir: Exported ONNX model directory when backend='onnx'
            search_backend: Vector backend name ('faiss' or 'pgvector')
            lean_hosted: Disable skill/company vector dependencies for hosted slices
        """
        self.db = open_database(db_path, ensure_schema=False)
        self.index_dir = Path(index_dir)
        self.embedding_backend = embedding_backend
        self.onnx_model_dir = Path(onnx_model_dir) if onnx_model_dir is not None else None
        self.model_version = resolve_model_version(model_version, embedding_backend)
        self.search_backend = (search_backend or os.environ.get("MCF_SEARCH_BACKEND") or "faiss").lower()
        env_lean_hosted = os.environ.get("MCF_LEAN_HOSTED", "").strip().lower() in {"1", "true", "yes", "on"}
        self.lean_hosted = lean_hosted or env_lean_hosted

        # Components (loaded lazily)
        if self.search_backend == "pgvector":
            self.vector_backend = PGVectorBackend(
                self.db,
                model_version=self.model_version,
                lean_hosted=self.lean_hosted,
            )
        else:
            self.vector_backend = FAISSVectorBackend(
                index_dir=self.index_dir,
                model_version=self.model_version,
            )
        self.generator = EmbeddingGenerator(
            model_name=model_version,
            backend=embedding_backend,
            onnx_model_dir=self.onnx_model_dir,
        )
        self.query_expander: Optional[QueryExpander] = None

        # Caches
        self._query_cache: TTLCache = TTLCache(
            maxsize=self.QUERY_CACHE_SIZE,
            ttl=self.QUERY_CACHE_TTL,
        )
        self._result_cache: TTLCache = TTLCache(
            maxsize=self.RESULT_CACHE_SIZE,
            ttl=self.RESULT_CACHE_TTL,
        )

        # State
        self._loaded = False
        self._degraded = False
        self._has_vector_index = False
        self._has_skill_clusters = False
        self._skill_vocabulary: Optional[list[tuple[str, str]]] = None

    def load(self) -> bool:
        """
        Load indexes and prepare for searching.

        Returns:
            True if indexes loaded successfully, False if degraded mode

        Note:
            This method is idempotent - safe to call multiple times.
        """
        if self._loaded:
            return not self._degraded

        logger.info("Loading semantic search engine...")

        # Try to load vector search backend
        try:
            if self.vector_backend.exists():
                self.vector_backend.load()
                self._has_vector_index = True
                logger.info("Vector backend loaded successfully (%s)", self.search_backend)
            else:
                logger.warning(
                    "Vector backend not found at %s. Run 'mcf embed-generate' to build indexes.",
                    self.index_dir,
                )
                self._degraded = True
        except IndexCompatibilityError as e:
            logger.warning(f"Index compatibility error: {e}. Falling back to keyword search.")
            self._degraded = True
        except Exception as e:
            logger.warning(f"Failed to load vector backend: {e}. Falling back to keyword search.")
            self._degraded = True

        # Try to load query expander (skill clusters)
        if not self.lean_hosted:
            try:
                self.query_expander = QueryExpander.load(self.index_dir)
                self._has_skill_clusters = True
                logger.info(f"Query expander loaded: {self.query_expander.get_stats()}")
            except FileNotFoundError:
                logger.warning(
                    "Skill clusters not found. Query expansion disabled. Run 'mcf embed-generate' to create clusters."
                )
            except Exception as e:
                logger.warning(f"Failed to load query expander: {e}")

        self._loaded = True
        return not self._degraded

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Main semantic search with all features.

        Flow:
        1. Check cache for identical request
        2. Apply SQL filters to get candidates
        3. Expand query if enabled
        4. Get query embedding (cached)
        5. Compute hybrid scores
        6. Return top k results

        Args:
            request: Search parameters

        Returns:
            SearchResponse with ranked results and metadata
        """
        start_time = time.time()

        # Ensure engine is loaded
        if not self._loaded:
            self.load()

        # Check result cache
        cache_key = request.cache_key()
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            cached.cache_hit = True
            return cached

        try:
            # Step 1: Get candidates
            # When SQL filters are active, they constrain the result set.
            # When unfiltered and vectors are available, search the full vector
            # backend directly to avoid loading millions of rows just to re-rank.
            has_filters = self._has_sql_filters(request)

            if has_filters:
                candidates = self._apply_sql_filters(request)
                total_candidates = len(candidates)
                candidate_uuids = [c["uuid"] for c in candidates]
            elif self._has_vector_index and not self._degraded:
                # Vector-first path: let the configured backend find the best
                # semantic matches from the full index, then score with BM25.
                vector_k = max(1000, request.limit * 50)
                query_embedding = self._get_query_embedding(request.query)
                semantic_results = self.vector_backend.search_jobs(
                    query_embedding,
                    k=vector_k,
                )
                candidate_uuids = [uuid for uuid, _ in semantic_results]
                total_candidates = self.vector_backend.total_jobs()
                if not candidate_uuids:
                    logger.warning("Vector backend returned no candidates; falling back to SQL candidate selection.")
                    candidates = self._apply_sql_filters(request)
                    total_candidates = len(candidates)
                    candidate_uuids = [c["uuid"] for c in candidates]
            else:
                # Degraded / no vectors: fall back to SQL with a generous cap
                candidates = self._apply_sql_filters(request)
                total_candidates = len(candidates)
                candidate_uuids = [c["uuid"] for c in candidates]

            if not candidate_uuids:
                response = SearchResponse(
                    results=[],
                    total_candidates=0,
                    search_time_ms=(time.time() - start_time) * 1000,
                    degraded=self._degraded,
                )
                return response

            # Step 2: Query expansion
            query_expansion = None
            search_query = request.query

            if request.expand_query and self.query_expander:
                expanded = self.query_expander.expand(request.query)
                if len(expanded) > 1:
                    query_expansion = expanded
                    # Use expanded terms for BM25 search
                    search_query = " ".join(expanded)
                    logger.debug(f"Query expanded: '{request.query}' -> {expanded}")

            # Step 3: Compute hybrid scores
            reference_skills = self._extract_skills_from_text(search_query)
            query_terms = query_expansion or self._query_terms(request.query)
            if self._has_vector_index and not self._degraded:
                scored_results, score_details = self._compute_hybrid_scores(
                    query=request.query,
                    search_query=search_query,
                    candidate_uuids=candidate_uuids,
                    alpha=request.alpha,
                    freshness_weight=request.freshness_weight,
                )
            else:
                # Degraded mode: keyword-only search
                scored_results, score_details = self._keyword_only_scores(
                    search_query=search_query,
                    candidate_uuids=candidate_uuids,
                    freshness_weight=request.freshness_weight,
                )

            # Step 4: Filter by minimum similarity and limit
            filtered_results = [(uuid, score) for uuid, score in scored_results if score >= request.min_similarity][
                : request.limit
            ]

            # Step 5: Enrich with full job data
            results = self._enrich_results(
                filtered_results,
                score_details=score_details,
                reference_skills=reference_skills,
                query_terms=query_terms,
            )

            response = SearchResponse(
                results=results,
                total_candidates=total_candidates,
                search_time_ms=(time.time() - start_time) * 1000,
                query_expansion=query_expansion,
                degraded=self._degraded,
                cache_hit=False,
            )

            # Cache result
            self._result_cache[cache_key] = response

            # Log analytics
            self._log_search(request, response)

            return response

        except (IndexNotBuiltError, IndexCompatibilityError) as e:
            logger.warning(f"Vector search failed: {e}. Falling back to keyword search.")
            self._degraded = True
            return self._keyword_fallback_search(request, start_time)

    def find_similar(self, request: SimilarJobsRequest) -> SearchResponse:
        """
        Find jobs similar to a given job.

        Uses the job's own embedding as the query vector.

        Args:
            request: Similar jobs request parameters

        Returns:
            SearchResponse with similar jobs
        """
        start_time = time.time()

        if not self._loaded:
            self.load()

        if self._degraded or not self._has_vector_index:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        # Get source job embedding
        source_embedding = self.db.get_embedding(request.job_uuid, "job")
        if source_embedding is None:
            logger.warning(f"No embedding found for job {request.job_uuid}")
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=self._degraded,
            )

        # Get source job for company exclusion
        source_job = self.db.get_job(request.job_uuid)

        # Search for similar (request extra to allow for filtering)
        search_k = request.limit + 10 if request.exclude_same_company else request.limit + 1

        try:
            results = self.vector_backend.search_jobs(source_embedding, k=search_k)
        except IndexNotBuiltError:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        # Filter results
        filtered: list[tuple[str, float]] = []
        score_details: dict[str, dict[str, float | list[str]]] = {}
        if request.exclude_same_company and source_job:
            candidate_uuids = [u for u, _ in results if u != request.job_uuid]
            jobs_bulk = self.db.get_jobs_bulk(candidate_uuids)
            source_company = source_job.get("company_name")
        else:
            jobs_bulk = {}
            source_company = None

        for uuid, score in results:
            # Skip the source job itself
            if uuid == request.job_uuid:
                continue

            # Skip same company if requested
            if source_company:
                job = jobs_bulk.get(uuid)
                if job and job.get("company_name") == source_company:
                    continue

            filtered.append((uuid, score))
            score_details[uuid] = {
                "semantic_score": round(float(score), 4),
            }

            if len(filtered) >= request.limit:
                break

        # Apply freshness boost
        if request.freshness_weight > 0 and filtered:
            freshness = self._compute_freshness_scores([uuid for uuid, _ in filtered])
            filtered = [(uuid, score + request.freshness_weight * freshness.get(uuid, 0.5)) for uuid, score in filtered]
            filtered.sort(key=lambda x: x[1], reverse=True)
            for uuid, total_score in filtered:
                detail = score_details.setdefault(uuid, {})
                detail["freshness_score"] = round(float(freshness.get(uuid, 0.5)), 4)
                detail["overall_fit"] = round(float(total_score), 4)

        reference_skills = self._parse_skills(source_job.get("skills")) if source_job else []
        query_terms = reference_skills or [request.job_uuid]

        results_enriched = self._enrich_results(
            filtered,
            score_details=score_details,
            reference_skills=reference_skills,
            query_terms=query_terms,
        )

        return SearchResponse(
            results=results_enriched,
            total_candidates=len(results_enriched),
            search_time_ms=(time.time() - start_time) * 1000,
            degraded=self._degraded,
        )

    def search_by_skill(self, request: SkillSearchRequest) -> SearchResponse:
        """
        Search jobs by skill similarity.

        Finds jobs that require skills similar to the specified skill.

        Args:
            request: Skill search parameters

        Returns:
            SearchResponse with matching jobs
        """
        start_time = time.time()

        if not self._loaded:
            self.load()

        # Generate embedding for the skill query
        skill_embedding = self._get_query_embedding(request.skill)

        if self._degraded or not self._has_vector_index:
            # Fall back to keyword search
            return self._keyword_fallback_search(
                SearchRequest(
                    query=request.skill,
                    limit=request.limit,
                    min_similarity=request.min_similarity,
                ),
                start_time,
            )

        try:
            has_sql_filters = any(
                [
                    request.salary_min is not None,
                    request.salary_max is not None,
                    request.employment_type is not None,
                ]
            )

            # Fetch more candidates when SQL filters will reduce the set
            k_multiplier = 4 if has_sql_filters else 2
            results = self.vector_backend.search_jobs(
                skill_embedding,
                k=request.limit * k_multiplier,
            )

            # Filter by minimum similarity
            filtered = [(uuid, score) for uuid, score in results if score >= request.min_similarity]

            # Apply SQL filters by intersecting with DB results
            if has_sql_filters and filtered:
                sql_filter_request = SearchRequest(
                    query="",
                    salary_min=request.salary_min,
                    salary_max=request.salary_max,
                    employment_type=request.employment_type,
                )
                sql_matches = self._apply_sql_filters(sql_filter_request)
                allowed_uuids = {job["uuid"] for job in sql_matches}
                filtered = [(uuid, score) for uuid, score in filtered if uuid in allowed_uuids]

            filtered = filtered[: request.limit]
            score_details = {
                uuid: {
                    "semantic_score": round(float(score), 4),
                    "overall_fit": round(float(score), 4),
                }
                for uuid, score in filtered
            }
            results_enriched = self._enrich_results(
                filtered,
                score_details=score_details,
                reference_skills=[request.skill],
                query_terms=[request.skill],
            )

            return SearchResponse(
                results=results_enriched,
                total_candidates=len(filtered),
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=self._degraded,
            )

        except IndexNotBuiltError:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

    def match_profile(
        self,
        profile_text: str,
        target_titles: Optional[list[str]] = None,
        salary_expectation_annual: Optional[int] = None,
        employment_type: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """
        Match a pasted candidate profile against job postings.

        Uses the profile embedding for semantic retrieval, then re-ranks with
        deterministic feature scoring so the UI can explain each match.
        """
        start_time = time.time()

        if not self._loaded:
            self.load()

        extracted_skills = self._extract_skills_from_text(profile_text)
        normalized_titles = [title.lower() for title in (target_titles or []) if title.strip()]
        profile_level = self._infer_profile_seniority(profile_text, target_titles or [])

        candidate_rows: list[tuple[str, float]] = []
        if self._has_vector_index and not self._degraded:
            profile_embedding = self._get_query_embedding(profile_text)
            search_k = max(limit * 30, 300)
            candidate_rows = self.vector_backend.search_jobs(profile_embedding, k=search_k)
        else:
            keyword_seed = " ".join(extracted_skills[:5]) or profile_text[:120]
            fallback = self._keyword_fallback_search(
                SearchRequest(
                    query=keyword_seed,
                    employment_type=employment_type,
                    region=region,
                    limit=max(limit * 10, 100),
                ),
                start_time,
            )
            candidate_rows = [(job.uuid, job.similarity_score) for job in fallback.results]

        jobs = self.db.get_jobs_bulk([uuid for uuid, _ in candidate_rows])
        scored: list[tuple[str, float]] = []
        score_details: dict[str, dict[str, float | list[str]]] = {}

        for uuid, raw_semantic in candidate_rows:
            job = jobs.get(uuid)
            if not job:
                continue

            if employment_type and job.get("employment_type") != employment_type:
                continue
            if region and job.get("region") != region:
                continue
            if normalized_titles and not any(title in (job.get("title", "").lower()) for title in normalized_titles):
                continue

            job_skills = self._parse_skills(job.get("skills"))
            matched_skills = sorted(set(extracted_skills).intersection(job_skills))
            missing_skills = sorted(set(extracted_skills) - set(job_skills))
            skill_overlap = len(matched_skills) / len(extracted_skills) if extracted_skills else 0.0

            retrieval_score = float(raw_semantic)
            semantic_score = retrieval_score if self._has_vector_index and not self._degraded else 0.0
            keyword_score = retrieval_score if self._degraded else 0.0
            seniority_fit = self._score_seniority(
                profile_level,
                self._normalize_seniority(job.get("seniority")),
            )
            salary_fit = self._score_salary_alignment(
                salary_expectation_annual,
                job.get("salary_annual_min"),
                job.get("salary_annual_max"),
            )

            weighted_parts = [
                (0.45, semantic_score if not self._degraded else keyword_score),
                (0.35, skill_overlap),
                (0.10, seniority_fit),
            ]
            if salary_expectation_annual is not None and salary_fit is not None:
                weighted_parts.append((0.10, salary_fit))

            total_weight = sum(weight for weight, _ in weighted_parts) or 1.0
            overall_fit = sum(weight * value for weight, value in weighted_parts) / total_weight

            scored.append((uuid, overall_fit))
            score_details[uuid] = {
                "semantic_score": round(float(semantic_score), 4) if not self._degraded else None,
                "bm25_score": round(float(keyword_score), 4) if self._degraded else None,
                "skill_overlap_score": round(float(skill_overlap), 4),
                "seniority_fit": round(float(seniority_fit), 4),
                "salary_fit": round(float(salary_fit), 4) if salary_fit is not None else None,
                "overall_fit": round(float(overall_fit), 4),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills[:10],
            }

        scored.sort(key=lambda item: item[1], reverse=True)
        limited = scored[:limit]
        results = self._enrich_results(
            limited,
            score_details=score_details,
            reference_skills=extracted_skills,
            query_terms=target_titles or extracted_skills[:8],
        )

        return {
            "results": results,
            "extracted_skills": extracted_skills,
            "total_candidates": len(scored),
            "search_time_ms": (time.time() - start_time) * 1000,
            "degraded": self._degraded,
        }

    def find_similar_companies(self, request: CompanySimilarityRequest) -> list[CompanySimilarity]:
        """
        Find companies with similar job profiles.

        Primary: Uses pre-computed multi-centroid company index for efficient matching.
        For each source centroid, searches the company index and aggregates by
        max similarity across all centroid pairs between source and target companies.

        Fallback: If company index is unavailable, computes a single centroid on-the-fly
        from the source company's job embeddings and searches the jobs index.

        Args:
            request: Company similarity request

        Returns:
            List of similar companies with stats
        """
        if not self._loaded:
            self.load()

        if self._degraded or not self._has_vector_index:
            return []

        # Check source company exists
        source_stats = self.db.get_company_stats(request.company_name)
        if source_stats.get("job_count", 0) == 0:
            logger.warning(f"No jobs found for company: {request.company_name}")
            return []

        # Primary path: use pre-computed company multi-centroid index
        if self.vector_backend.has_company_index():
            source_centroids = self.vector_backend.get_company_centroids(request.company_name)
            if source_centroids is not None:
                return self._find_similar_companies_multi_centroid(request, source_centroids)

        # Fallback: on-the-fly single centroid via jobs index
        return self._find_similar_companies_fallback(request)

    def _find_similar_companies_multi_centroid(
        self,
        request: CompanySimilarityRequest,
        source_centroids: np.ndarray,
    ) -> list[CompanySimilarity]:
        """
        Find similar companies using multi-centroid matching.

        For each source centroid, searches the company index. For each target
        company, takes the max similarity across all source-target centroid pairs.

        Args:
            request: Company similarity request
            source_centroids: Source company's centroid array (n_centroids, dim)

        Returns:
            List of CompanySimilarity results
        """
        company_scores: dict[str, float] = {}

        # For each source centroid, search company index
        for centroid in source_centroids:
            try:
                results = self.vector_backend.search_companies(centroid, k=request.limit + 10)
            except IndexNotBuiltError:
                continue

            for company_name, score in results:
                if company_name == request.company_name:
                    continue
                # Max similarity across all centroid pairs
                if company_name not in company_scores or score > company_scores[company_name]:
                    company_scores[company_name] = score

        # Sort by score descending
        sorted_companies = sorted(company_scores.items(), key=lambda x: x[1], reverse=True)[: request.limit]

        # Enrich with company stats
        results_list: list[CompanySimilarity] = []
        for company_name, score in sorted_companies:
            stats = self.db.get_company_stats(company_name)
            results_list.append(
                CompanySimilarity(
                    company_name=company_name,
                    similarity_score=score,
                    job_count=stats.get("job_count", 0),
                    avg_salary=stats.get("avg_salary"),
                    top_skills=stats.get("top_skills", [])[:5],
                )
            )

        return results_list

    def _find_similar_companies_fallback(self, request: CompanySimilarityRequest) -> list[CompanySimilarity]:
        """
        Fallback: compute single centroid on-the-fly and search jobs index.

        Used when company index is not available (degraded mode).

        Args:
            request: Company similarity request

        Returns:
            List of CompanySimilarity results
        """
        company_jobs = self.db.search_jobs(
            company_name=request.company_name,
            limit=50,
        )

        if not company_jobs:
            return []

        job_uuids = [j["uuid"] for j in company_jobs]
        embeddings_dict = self.db.get_embeddings_for_uuids(job_uuids)

        if not embeddings_dict:
            return []

        # Compute single centroid
        embeddings = np.array(list(embeddings_dict.values()))
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        try:
            similar_jobs = self.vector_backend.search_jobs(centroid, k=500)
        except IndexNotBuiltError:
            return []

        # Aggregate by company
        jobs_bulk = self.db.get_jobs_bulk([u for u, _ in similar_jobs])
        company_scores: dict[str, list[float]] = {}
        for uuid, score in similar_jobs:
            job = jobs_bulk.get(uuid)
            if job and job.get("company_name"):
                company = job["company_name"]
                if company == request.company_name:
                    continue
                if company not in company_scores:
                    company_scores[company] = []
                company_scores[company].append(score)

        # Take average score per company
        company_avg: list[tuple[str, float]] = [
            (company, sum(scores) / len(scores)) for company, scores in company_scores.items()
        ]
        company_avg.sort(key=lambda x: x[1], reverse=True)

        # Enrich with company stats
        results: list[CompanySimilarity] = []
        for company_name, score in company_avg[: request.limit]:
            stats = self.db.get_company_stats(company_name)
            results.append(
                CompanySimilarity(
                    company_name=company_name,
                    similarity_score=score,
                    job_count=stats.get("job_count", 0),
                    avg_salary=stats.get("avg_salary"),
                    top_skills=stats.get("top_skills", [])[:5],
                )
            )

        return results

    def get_stats(self) -> dict:
        """
        Get search engine statistics.

        Returns:
            Dict with index stats, cache stats, and engine state
        """
        stats = {
            "loaded": self._loaded,
            "degraded": self._degraded,
            "has_vector_index": self._has_vector_index,
            "has_skill_clusters": self._has_skill_clusters,
            "model_version": self.model_version,
            "embedding_backend": self.embedding_backend,
            "index_dir": str(self.index_dir),
        }

        # Cache stats
        stats["caches"] = {
            "query_cache_size": len(self._query_cache),
            "query_cache_max": self.QUERY_CACHE_SIZE,
            "result_cache_size": len(self._result_cache),
            "result_cache_max": self.RESULT_CACHE_SIZE,
        }

        # Index stats (if loaded)
        if self._has_vector_index:
            try:
                stats["index_stats"] = self.vector_backend.get_stats()
            except Exception as e:
                stats["index_stats"] = {"error": str(e)}

        # Query expander stats
        if self.query_expander:
            stats["query_expander"] = self.query_expander.get_stats()

        # Database stats
        stats["embedding_stats"] = self.db.get_embedding_stats()

        return stats

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _has_sql_filters(self, request: SearchRequest) -> bool:
        """Check whether a request has any active SQL filter parameters."""
        return any(
            [
                request.salary_min is not None,
                request.salary_max is not None,
                request.employment_type is not None,
                request.company is not None,
                request.region is not None,
            ]
        )

    def _apply_sql_filters(self, request: SearchRequest) -> list[dict]:
        """
        Apply SQL filters and return matching jobs.

        When filters are active, returns all matching rows (no hard cap)
        since the filters already constrain the result set. When no filters
        are present, applies a cap to avoid loading the entire database.

        Args:
            request: Search request with filter parameters

        Returns:
            List of job dicts matching filters
        """
        # When filters are active, they constrain the result set — no cap needed.
        # When unfiltered, cap at 500K to bound memory while covering most datasets.
        limit = 10_000_000 if self._has_sql_filters(request) else 500_000
        return self.db.search_jobs(
            salary_min=request.salary_min,
            salary_max=request.salary_max,
            employment_type=request.employment_type,
            company_name=request.company,
            region=request.region,
            limit=limit,
        )

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query string (with caching).

        Args:
            query: Query text

        Returns:
            Embedding array of shape (dimension,)
        """
        # Check cache
        cache_key = f"query:{query}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = self.generator.backend.encode_one(
            query,
            normalize_embeddings=True,
        )

        # Cache it
        self._query_cache[cache_key] = embedding

        return embedding

    def _compute_hybrid_scores(
        self,
        query: str,
        search_query: str,
        candidate_uuids: list[str],
        alpha: float,
        freshness_weight: float,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, float | list[str]]]]:
        """
        Compute hybrid scores combining semantic and keyword search.

        Formula: score = alpha * semantic + (1 - alpha) * bm25 + freshness_boost

        Args:
            query: Original query (for semantic embedding)
            search_query: Expanded query (for BM25)
            candidate_uuids: UUIDs to score
            alpha: Weight for semantic vs keyword (1.0 = semantic only)
            freshness_weight: Weight for freshness boost

        Returns:
            List of (uuid, score) tuples, sorted by score descending
        """
        candidate_set = set(candidate_uuids)

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Get semantic scores via filtered vector search
        try:
            semantic_results = self.vector_backend.search_jobs_filtered(
                query_embedding,
                candidate_uuids=list(candidate_set),
                k=len(candidate_uuids),
            )
            semantic_scores = {uuid: score for uuid, score in semantic_results}
        except IndexNotBuiltError:
            semantic_scores = {}

        # Get BM25 scores
        bm25_scores = self._get_bm25_scores(search_query, candidate_uuids)

        # Get freshness scores
        freshness_scores = self._compute_freshness_scores(candidate_uuids) if freshness_weight > 0 else {}

        # Normalize scores to [0, 1] range for fair combination
        semantic_scores = self._normalize_scores(semantic_scores)
        bm25_scores = self._normalize_scores(bm25_scores)

        # Combine scores
        combined: dict[str, float] = {}
        details: dict[str, dict[str, float | list[str]]] = {}
        for uuid in candidate_uuids:
            sem_score = semantic_scores.get(uuid, 0.0)
            bm25_score = bm25_scores.get(uuid, 0.0)
            fresh_score = freshness_scores.get(uuid, 0.5)

            combined[uuid] = alpha * sem_score + (1 - alpha) * bm25_score + freshness_weight * fresh_score
            details[uuid] = {
                "semantic_score": round(float(sem_score), 4),
                "bm25_score": round(float(bm25_score), 4),
                "freshness_score": round(float(fresh_score), 4),
                "overall_fit": round(float(combined[uuid]), 4),
            }

        # Sort by score descending
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        return sorted_results, details

    def _get_bm25_scores(self, query: str, candidate_uuids: list[str]) -> dict[str, float]:
        """
        Get BM25 scores for candidates.

        Uses filtered BM25 search that restricts scoring to the candidate set,
        ensuring no relevant candidate is missed due to global ranking cutoffs.

        Args:
            query: Search query
            candidate_uuids: UUIDs to score

        Returns:
            Dict mapping uuid -> BM25 score (higher = more relevant)
        """
        try:
            candidate_set = set(candidate_uuids)
            # BM25 returns negative scores (lower = better)
            bm25_results = self.db.bm25_search_filtered(query, candidate_set)

            # Negate to make higher = better
            scores = {}
            for uuid, score in bm25_results:
                scores[uuid] = -score

            return scores
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return {}

    def _keyword_only_scores(
        self,
        search_query: str,
        candidate_uuids: list[str],
        freshness_weight: float,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, float | list[str]]]]:
        """
        Compute scores using only keyword (BM25) search.

        Used when vector index is unavailable (degraded mode).

        Args:
            search_query: Search query
            candidate_uuids: UUIDs to score
            freshness_weight: Weight for freshness boost

        Returns:
            List of (uuid, score) tuples
        """
        bm25_scores = self._get_bm25_scores(search_query, candidate_uuids)
        bm25_scores = self._normalize_scores(bm25_scores)

        freshness_scores = self._compute_freshness_scores(candidate_uuids) if freshness_weight > 0 else {}

        combined: dict[str, float] = {}
        details: dict[str, dict[str, float | list[str]]] = {}
        for uuid in candidate_uuids:
            bm25_score = bm25_scores.get(uuid, 0.0)
            fresh_score = freshness_scores.get(uuid, 0.5)
            combined[uuid] = bm25_score + freshness_weight * fresh_score
            details[uuid] = {
                "bm25_score": round(float(bm25_score), 4),
                "freshness_score": round(float(fresh_score), 4),
                "overall_fit": round(float(combined[uuid]), 4),
            }

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results, details

    def _keyword_fallback_search(self, request: SearchRequest, start_time: float) -> SearchResponse:
        """
        Fallback search using only keywords when vectors fail.

        Args:
            request: Search request
            start_time: Time when search started

        Returns:
            SearchResponse with keyword-only results
        """
        candidates = self._apply_sql_filters(request)

        if not candidates:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        scored_results, score_details = self._keyword_only_scores(
            search_query=request.query,
            candidate_uuids=[c["uuid"] for c in candidates],
            freshness_weight=request.freshness_weight,
        )

        filtered_results = scored_results[: request.limit]
        results = self._enrich_results(
            filtered_results,
            score_details=score_details,
            reference_skills=self._extract_skills_from_text(request.query),
            query_terms=self._query_terms(request.query),
        )

        return SearchResponse(
            results=results,
            total_candidates=len(candidates),
            search_time_ms=(time.time() - start_time) * 1000,
            degraded=True,
        )

    def _compute_freshness_scores(self, uuids: list[str]) -> dict[str, float]:
        """
        Compute freshness scores based on posting date.

        More recent jobs get higher scores (0-1 range).

        Args:
            uuids: Job UUIDs to score

        Returns:
            Dict mapping uuid -> freshness score
        """
        scores: dict[str, float] = {}
        today = date.today()

        jobs = self.db.get_jobs_bulk(uuids)
        for uuid in uuids:
            job = jobs.get(uuid)
            if job and job.get("posted_date"):
                try:
                    posted = job["posted_date"]
                    if isinstance(posted, str):
                        posted = date.fromisoformat(posted)

                    days_ago = (today - posted).days
                    # Exponential decay: 30-day half-life
                    # Fresh jobs (0 days) = 1.0, 30 days = 0.5, 60 days = 0.25
                    scores[uuid] = np.exp(-days_ago / 30 * np.log(2))
                except (ValueError, TypeError):
                    scores[uuid] = 0.5
            else:
                scores[uuid] = 0.5

        return scores

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: Dict mapping uuid -> raw score

        Returns:
            Dict mapping uuid -> normalized score
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            # All scores are the same
            return {uuid: 0.5 for uuid in scores}

        return {uuid: (score - min_val) / (max_val - min_val) for uuid, score in scores.items()}

    def _load_skill_vocabulary(self) -> list[tuple[str, str]]:
        """Load and cache the canonical skill vocabulary for extraction."""
        if self._skill_vocabulary is None:
            skills = self.db.get_all_unique_skills()
            self._skill_vocabulary = [
                (skill, skill.lower()) for skill in sorted(skills, key=len, reverse=True) if len(skill.strip()) >= 2
            ]
        return self._skill_vocabulary

    @staticmethod
    def _parse_skills(skills: Optional[str]) -> list[str]:
        """Split a comma-separated skills string into normalized labels."""
        if not skills:
            return []
        return [skill.strip() for skill in skills.split(",") if skill.strip()]

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        """Extract a small set of readable query terms for explanations."""
        terms = [term for term in re.split(r"\W+", query) if len(term) >= 3]
        return terms[:8]

    def _extract_skills_from_text(self, text: str, limit: int = 12) -> list[str]:
        """Deterministically extract known skill names from free text."""
        normalized_text = f" {text.lower()} "
        matches: list[str] = []
        for skill, lowered in self._load_skill_vocabulary():
            pattern = rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])"
            if re.search(pattern, normalized_text):
                matches.append(skill)
                if len(matches) >= limit:
                    break
        return matches

    @staticmethod
    def _normalize_seniority(raw: Optional[str]) -> Optional[str]:
        """Map free-form seniority labels into a small ordered vocabulary."""
        if not raw:
            return None
        value = raw.lower()
        mapping = {
            "intern": ["intern", "attachment", "trainee"],
            "junior": ["junior", "entry", "associate", "executive"],
            "mid": ["mid", "experienced", "specialist"],
            "senior": ["senior", "lead", "manager", "principal"],
            "director": ["director", "head", "vp", "vice president"],
        }
        for level, keywords in mapping.items():
            if any(keyword in value for keyword in keywords):
                return level
        return None

    def _infer_profile_seniority(self, profile_text: str, target_titles: Optional[list[str]] = None) -> Optional[str]:
        """Infer approximate seniority from the pasted profile text."""
        combined = " ".join((target_titles or []) + [profile_text])
        return self._normalize_seniority(combined)

    @staticmethod
    def _score_seniority(profile_level: Optional[str], job_level: Optional[str]) -> float:
        """Score how close the candidate's inferred level is to the job level."""
        if profile_level is None or job_level is None:
            return 0.5

        order = ["intern", "junior", "mid", "senior", "director"]
        profile_idx = order.index(profile_level)
        job_idx = order.index(job_level)
        distance = abs(profile_idx - job_idx)
        if distance == 0:
            return 1.0
        if distance == 1:
            return 0.7
        if distance == 2:
            return 0.35
        return 0.1

    @staticmethod
    def _score_salary_alignment(
        expectation_annual: Optional[int],
        salary_annual_min: Optional[int],
        salary_annual_max: Optional[int],
    ) -> Optional[float]:
        """Score how well a job salary range aligns to an annual expectation."""
        if expectation_annual is None:
            return None

        if salary_annual_min is None and salary_annual_max is None:
            return 0.5

        low = salary_annual_min if salary_annual_min is not None else salary_annual_max
        high = salary_annual_max if salary_annual_max is not None else salary_annual_min
        assert low is not None and high is not None

        if low <= expectation_annual <= high:
            return 1.0
        if expectation_annual < low:
            gap = (low - expectation_annual) / max(low, 1)
        else:
            gap = (expectation_annual - high) / max(expectation_annual, 1)
        return max(0.0, round(1.0 - gap, 4))

    def _enrich_results(
        self,
        scored_results: list[tuple[str, float]],
        score_details: Optional[dict[str, dict[str, float | list[str]]]] = None,
        reference_skills: Optional[list[str]] = None,
        query_terms: Optional[list[str]] = None,
    ) -> list[JobResult]:
        """
        Convert (uuid, score) tuples to full JobResult objects.

        Args:
            scored_results: List of (uuid, score) tuples

        Returns:
            List of JobResult objects with full job data
        """
        results: list[JobResult] = []

        jobs = self.db.get_jobs_bulk([uuid for uuid, _ in scored_results])
        reference_set = set(reference_skills or [])
        for uuid, score in scored_results:
            job = jobs.get(uuid)
            if job:
                details = score_details.get(uuid, {}) if score_details else {}

                # Parse posted_date if it's a string
                posted_date = job.get("posted_date")
                if isinstance(posted_date, str):
                    try:
                        posted_date = date.fromisoformat(posted_date)
                    except ValueError:
                        posted_date = None

                # Truncate description for response
                description = job.get("description", "") or ""
                if len(description) > 500:
                    description = description[:500] + "..."

                job_skills = self._parse_skills(job.get("skills"))
                matched_skills = sorted(reference_set.intersection(job_skills))
                missing_skills = sorted(reference_set - set(job_skills))

                explanation = SearchExplanation(
                    semantic_score=details.get("semantic_score"),
                    bm25_score=details.get("bm25_score"),
                    freshness_score=details.get("freshness_score"),
                    matched_skills=details.get("matched_skills", matched_skills),
                    missing_skills=details.get("missing_skills", missing_skills[:10]),
                    query_terms=query_terms or [],
                    skill_overlap_score=details.get("skill_overlap_score"),
                    seniority_fit=details.get("seniority_fit"),
                    salary_fit=details.get("salary_fit"),
                    overall_fit=details.get("overall_fit", round(float(score), 4)),
                )

                results.append(
                    JobResult(
                        uuid=job["uuid"],
                        title=job.get("title", ""),
                        company_name=job.get("company_name", ""),
                        description=description,
                        salary_min=job.get("salary_min"),
                        salary_max=job.get("salary_max"),
                        employment_type=job.get("employment_type"),
                        seniority=job.get("seniority"),
                        skills=job.get("skills"),
                        location=job.get("location"),
                        posted_date=posted_date,
                        job_url=job.get("job_url"),
                        similarity_score=score,
                        semantic_score=details.get("semantic_score"),
                        bm25_score=details.get("bm25_score"),
                        freshness_score=details.get("freshness_score"),
                        matched_skills=explanation.matched_skills,
                        missing_skills=explanation.missing_skills,
                        explanations=explanation,
                    )
                )

        return results

    def _log_search(self, request: SearchRequest, response: SearchResponse) -> None:
        """
        Log search to analytics table.

        Args:
            request: Search request
            response: Search response
        """
        try:
            query_type = "hybrid" if not self._degraded else "keyword"

            self.db.log_search(
                query=request.query,
                query_type=query_type,
                result_count=len(response.results),
                latency_ms=response.search_time_ms,
                cache_hit=response.cache_hit,
                degraded=response.degraded,
                filters_used={
                    "salary_min": request.salary_min,
                    "salary_max": request.salary_max,
                    "employment_type": request.employment_type,
                    "company": request.company,
                    "region": request.region,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log search analytics: {e}")

    def get_skill_cloud(self, min_jobs: int = 10, limit: int = 100) -> dict:
        """
        Get skill frequency data for visualization (word cloud, bar chart).

        Fetches skill occurrence counts from the database and annotates
        each skill with its cluster ID from the query expander (if loaded).

        Args:
            min_jobs: Minimum job count for a skill to be included
            limit: Maximum number of skills to return

        Returns:
            Dict with 'items' (list of skill dicts) and 'total_unique_skills'
        """
        if not self._loaded:
            self.load()

        # Single DB call — get all skills, then filter in Python
        all_skills = self.db.get_skill_frequencies(min_jobs=1, limit=100000)
        total_unique = len(all_skills)

        # Apply caller's min_jobs filter (already sorted by freq desc)
        filtered = [(s, c) for s, c in all_skills if c >= min_jobs][:limit]

        items = []
        for skill, count in filtered:
            cluster_id = None
            if self.query_expander:
                cluster_id = self.query_expander.skill_to_cluster.get(skill)
            items.append(
                {
                    "skill": skill,
                    "job_count": count,
                    "cluster_id": cluster_id,
                }
            )

        return {
            "items": items,
            "total_unique_skills": total_unique,
        }

    def get_related_skills(self, skill: str, k: int = 10) -> Optional[dict]:
        """
        Get skills related to a given skill.

        Primary strategy: embedding similarity on the configured skill index,
        annotated with cluster membership from QueryExpander.
        Fallback: cluster-only lookup when vector skill indexes are unavailable.

        Args:
            skill: Skill name to find relatives for
            k: Maximum number of related skills to return

        Returns:
            Dict with 'skill' and 'related' list, or None if skill not found
        """
        if not self._loaded:
            self.load()

        source_cluster = None
        if self.query_expander:
            source_cluster = self.query_expander.skill_to_cluster.get(skill)

        # Primary: embedding similarity through the configured vector backend
        if self._has_vector_index and self.vector_backend.has_skill_index():
            skill_embedding = self.vector_backend.get_skill_embedding(skill)
            if skill_embedding is not None:
                similar = self.vector_backend.search_skills(skill_embedding, k=k + 1)

                related = []
                for s, score in similar:
                    if s == skill:
                        continue
                    same_cluster = False
                    if self.query_expander:
                        s_cluster = self.query_expander.skill_to_cluster.get(s)
                        same_cluster = s_cluster is not None and s_cluster == source_cluster
                    related.append(
                        {
                            "skill": s,
                            "similarity": round(float(score), 4),
                            "same_cluster": same_cluster,
                        }
                    )

                return {
                    "skill": skill,
                    "related": related[:k],
                }

        # Fallback: cluster-only (no similarity scores available)
        if self.query_expander:
            cluster_skills = self.query_expander.get_related_skills(skill, k=k)
            if cluster_skills:
                return {
                    "skill": skill,
                    "related": [{"skill": s, "similarity": 1.0, "same_cluster": True} for s in cluster_skills],
                }

        return None  # Skill not found

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._query_cache.clear()
        self._result_cache.clear()
        logger.info("Search engine caches cleared")
