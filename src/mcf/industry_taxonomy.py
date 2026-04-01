"""
Conservative industry taxonomy and title-family normalization helpers.

This module intentionally optimizes for trust over coverage. It provides:
- deterministic category normalization for noisy MCF category strings
- a small two-level industry taxonomy (sector + subsector)
- explicit fallback ordering for industry classification
- title-family normalization for same-role and adjacent-role reasoning
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

UNKNOWN_SECTOR = "unknown"
UNKNOWN_SUBSECTOR = "unknown"

DIRECT_CATEGORY_CONFIDENCE = 0.9
COMPANY_DOMINANT_CONFIDENCE = 0.65
SKILL_AFFINITY_CONFIDENCE = 0.55

COMPANY_DOMINANT_MIN_COUNT = 3
COMPANY_DOMINANT_MIN_SHARE = 0.6
SKILL_AFFINITY_MIN_MATCHES = 2

_CATEGORY_SPLIT_RE = re.compile(r"\s*(?:[;/|]|,)\s*", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SENIORITY_RE = re.compile(
    r"\b("
    r"senior|sr|junior|jr|principal|staff|head|chief|director|"
    r"assistant|associate|intern|trainee|entry[\s-]*level|mid[\s-]*level|"
    r"ii|iii|iv|v"
    r")\b",
    re.IGNORECASE,
)


class IndustrySource(str, Enum):
    """Provenance of an industry assignment."""

    DIRECT_CATEGORY = "direct_category"
    COMPANY_DOMINANT = "company_dominant"
    SKILL_AFFINITY = "skill_affinity"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IndustryClassification:
    """
    Two-level industry assignment with explicit provenance and confidence.
    """

    sector: str = UNKNOWN_SECTOR
    subsector: str = UNKNOWN_SUBSECTOR
    source: IndustrySource = IndustrySource.UNKNOWN
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    normalized_categories: tuple[str, ...] = ()

    @property
    def is_unknown(self) -> bool:
        """Whether the job remains deliberately unclassified."""
        return self.sector == UNKNOWN_SECTOR


@dataclass(frozen=True)
class TitleFamily:
    """
    Normalized title family for role comparisons.

    `canonical` is stable and suitable for cache keys or grouping.
    `tokens` keep the normalized vocabulary available for adjacency checks.
    """

    canonical: str
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndustryTaxonomyConfig:
    """Thresholds for conservative fallback behavior."""

    company_dominant_min_count: int = COMPANY_DOMINANT_MIN_COUNT
    company_dominant_min_share: float = COMPANY_DOMINANT_MIN_SHARE
    skill_affinity_min_matches: int = SKILL_AFFINITY_MIN_MATCHES


_CATEGORY_TO_BUCKET = {
    "information technology": ("technology", "software_and_platforms"),
    "software development": ("technology", "software_and_platforms"),
    "cloud computing": ("technology", "cloud_and_infrastructure"),
    "data science": ("technology", "data_and_ai"),
    "artificial intelligence": ("technology", "data_and_ai"),
    "cybersecurity": ("technology", "cybersecurity"),
    "engineering": ("engineering", "general_engineering"),
    "manufacturing": ("engineering", "manufacturing"),
    "construction": ("engineering", "built_environment"),
    "banking": ("financial_services", "banking"),
    "finance": ("financial_services", "banking"),
    "insurance": ("financial_services", "insurance"),
    "accounting": ("financial_services", "accounting"),
    "healthcare": ("healthcare", "clinical_and_care_delivery"),
    "health care": ("healthcare", "clinical_and_care_delivery"),
    "biotechnology": ("healthcare", "biotech_and_life_sciences"),
    "pharmaceutical": ("healthcare", "biotech_and_life_sciences"),
    "sales": ("commercial", "sales_and_business_development"),
    "marketing": ("commercial", "marketing_and_brand"),
    "retail": ("commercial", "retail_and_ecommerce"),
    "e-commerce": ("commercial", "retail_and_ecommerce"),
    "human resources": ("corporate_services", "people_and_operations"),
    "operations": ("corporate_services", "people_and_operations"),
    "legal": ("corporate_services", "legal_and_compliance"),
    "education": ("education", "teaching_and_learning"),
    "training": ("education", "teaching_and_learning"),
    "logistics": ("supply_chain", "logistics_and_warehousing"),
    "supply chain": ("supply_chain", "logistics_and_warehousing"),
}

_SKILL_TO_BUCKET = {
    "python": ("technology", "data_and_ai"),
    "machine learning": ("technology", "data_and_ai"),
    "artificial intelligence": ("technology", "data_and_ai"),
    "tensorflow": ("technology", "data_and_ai"),
    "pytorch": ("technology", "data_and_ai"),
    "sql": ("technology", "data_and_ai"),
    "aws": ("technology", "cloud_and_infrastructure"),
    "azure": ("technology", "cloud_and_infrastructure"),
    "gcp": ("technology", "cloud_and_infrastructure"),
    "kubernetes": ("technology", "cloud_and_infrastructure"),
    "docker": ("technology", "cloud_and_infrastructure"),
    "react": ("technology", "software_and_platforms"),
    "javascript": ("technology", "software_and_platforms"),
    "typescript": ("technology", "software_and_platforms"),
    "node.js": ("technology", "software_and_platforms"),
    "salesforce": ("commercial", "sales_and_business_development"),
    "crm": ("commercial", "sales_and_business_development"),
    "ledger": ("financial_services", "accounting"),
    "audit": ("financial_services", "accounting"),
    "patient care": ("healthcare", "clinical_and_care_delivery"),
    "nursing": ("healthcare", "clinical_and_care_delivery"),
}

_TOKEN_REPLACEMENTS = {
    "software": "software",
    "developer": "engineer",
    "development": "engineer",
    "programmer": "engineer",
    "engineering": "engineer",
    "ml": "machine",
    "qa": "quality assurance",
    "ux": "design",
    "ui": "design",
    "frontend": "front end",
    "back-end": "back end",
    "backend": "back end",
    "fullstack": "full stack",
    "devops": "platform",
    "sre": "platform",
    "analyst": "analyst",
}

_NOISE_TOKENS = {
    "and",
    "the",
    "of",
    "for",
    "to",
    "a",
    "an",
    "specialist",
    "specialists",
}

_CANONICAL_TITLE_PATTERNS = [
    (re.compile(r"\b(machine|ai)\s+learning\s+engineer\b"), ("machine", "learning", "engineer")),
    (re.compile(r"\bdata\s+scientist\b"), ("data", "scientist")),
    (re.compile(r"\bdata\s+analyst\b"), ("data", "analyst")),
    (re.compile(r"\b(product)\s+manager\b"), ("product", "manager")),
    (re.compile(r"\b(back\s+end|backend)\s+(developer|engineer)\b"), ("software", "engineer")),
    (re.compile(r"\b(front\s+end|frontend)\s+(developer|engineer)\b"), ("software", "engineer")),
    (re.compile(r"\b(software|full stack|front end|back end)\s+engineer\b"), ("software", "engineer")),
    (re.compile(r"\b(platform|site reliability)\s+engineer\b"), ("platform", "engineer")),
]

_ADJACENT_ROLE_SUFFIX_PAIRS = {
    frozenset({"scientist", "analyst"}),
    frozenset({"engineer", "architect"}),
    frozenset({"engineer", "developer"}),
    frozenset({"manager", "lead"}),
}


def _slugify(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", value.lower()).strip()


def normalize_category_string(raw_category: str) -> list[str]:
    """Normalize a single category string, expanding common multi-value separators."""
    if not raw_category:
        return []

    normalized: list[str] = []
    for part in _CATEGORY_SPLIT_RE.split(raw_category):
        slug = _slugify(part)
        if slug and slug not in normalized:
            normalized.append(slug)
    return normalized


def normalize_categories(raw_categories: Iterable[str]) -> list[str]:
    """Normalize one or many category strings into a deterministic unique list."""
    normalized: list[str] = []
    for category in raw_categories:
        for part in normalize_category_string(category):
            if part not in normalized:
                normalized.append(part)
    return normalized


def classify_industry(
    raw_categories: Iterable[str],
    *,
    company_classifications: Optional[Iterable[IndustryClassification]] = None,
    skills: Optional[Iterable[str]] = None,
    config: IndustryTaxonomyConfig = IndustryTaxonomyConfig(),
) -> IndustryClassification:
    """
    Classify a job using direct category mapping, then company inference,
    then skill affinity. Unknown stays unknown if evidence is weak.
    """
    normalized_categories = tuple(normalize_categories(raw_categories))

    direct = _classify_direct(normalized_categories)
    if direct is not None:
        return direct

    company = infer_company_dominant_industry(company_classifications or [], config=config)
    if company is not None:
        return IndustryClassification(
            sector=company.sector,
            subsector=company.subsector,
            source=IndustrySource.COMPANY_DOMINANT,
            confidence=COMPANY_DOMINANT_CONFIDENCE,
            evidence=company.evidence,
            normalized_categories=normalized_categories,
        )

    skill_based = _classify_from_skills(skills or [], config=config)
    if skill_based is not None:
        return IndustryClassification(
            sector=skill_based.sector,
            subsector=skill_based.subsector,
            source=IndustrySource.SKILL_AFFINITY,
            confidence=SKILL_AFFINITY_CONFIDENCE,
            evidence=skill_based.evidence,
            normalized_categories=normalized_categories,
        )

    return IndustryClassification(normalized_categories=normalized_categories)


def classification_from_bucket(bucket: str | None) -> IndustryClassification:
    """Parse a persisted `sector/subsector` bucket into an IndustryClassification."""
    normalized = (bucket or "").strip().lower()
    if not normalized or "/" not in normalized:
        return IndustryClassification()

    sector, subsector = normalized.split("/", 1)
    if not sector or not subsector:
        return IndustryClassification()
    return IndustryClassification(sector=sector, subsector=subsector)


def infer_company_dominant_industry(
    company_classifications: Iterable[IndustryClassification],
    *,
    config: IndustryTaxonomyConfig = IndustryTaxonomyConfig(),
) -> Optional[IndustryClassification]:
    """Infer a company's dominant industry when enough non-unknown evidence exists."""
    known = [classification for classification in company_classifications if not classification.is_unknown]
    if len(known) < config.company_dominant_min_count:
        return None

    counts = Counter((classification.sector, classification.subsector) for classification in known)
    (sector, subsector), count = counts.most_common(1)[0]
    share = count / len(known)
    if share < config.company_dominant_min_share:
        return None

    return IndustryClassification(
        sector=sector,
        subsector=subsector,
        source=IndustrySource.COMPANY_DOMINANT,
        confidence=share,
        evidence=(f"company_share:{share:.2f}", f"sample_size:{len(known)}"),
    )


def normalize_title_family(title: str) -> TitleFamily:
    """Normalize noisy titles into stable role families."""
    normalized = _slugify(title)

    for pattern, tokens in _CANONICAL_TITLE_PATTERNS:
        if pattern.search(normalized):
            return TitleFamily(canonical="-".join(tokens), tokens=tokens)

    normalized = _SENIORITY_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens: list[str] = []
    for raw_token in normalized.split():
        replacement = _TOKEN_REPLACEMENTS.get(raw_token, raw_token)
        for token in replacement.split():
            if token and token not in _NOISE_TOKENS:
                tokens.append(token)

    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)

    if not deduped:
        return TitleFamily(canonical="unknown", tokens=("unknown",))

    if "engineer" in deduped:
        anchor = next((token for token in deduped if token != "engineer"), "general")
        return TitleFamily(canonical=f"{anchor}-engineer", tokens=tuple(deduped))

    return TitleFamily(canonical="-".join(deduped), tokens=tuple(deduped))


def is_same_role(title_a: str, title_b: str) -> bool:
    """Whether two titles normalize to the same family."""
    return normalize_title_family(title_a).canonical == normalize_title_family(title_b).canonical


def is_adjacent_role(title_a: str, title_b: str) -> bool:
    """Whether two titles are close but not identical role families."""
    family_a = normalize_title_family(title_a)
    family_b = normalize_title_family(title_b)
    if family_a.canonical == family_b.canonical:
        return False

    overlap = set(family_a.tokens) & set(family_b.tokens)
    if not overlap:
        return False

    parts_a = family_a.canonical.split("-")
    parts_b = family_b.canonical.split("-")
    prefix_a = "-".join(parts_a[:-1])
    prefix_b = "-".join(parts_b[:-1])
    suffix_pair = frozenset({parts_a[-1], parts_b[-1]})

    return (prefix_a and prefix_a == prefix_b and suffix_pair in _ADJACENT_ROLE_SUFFIX_PAIRS) or len(overlap) >= 2


def industry_distance(left: IndustryClassification, right: IndustryClassification) -> int:
    """
    Conservative distance metric for downstream pivot logic.

    0 = same subsector
    1 = same sector, different subsector
    2 = different known sectors
    3 = one or both sides unknown
    """
    if left.is_unknown or right.is_unknown:
        return 3
    if left.sector == right.sector and left.subsector == right.subsector:
        return 0
    if left.sector == right.sector:
        return 1
    return 2


def _classify_direct(normalized_categories: tuple[str, ...]) -> Optional[IndustryClassification]:
    for category in normalized_categories:
        bucket = _CATEGORY_TO_BUCKET.get(category)
        if bucket is None:
            continue
        sector, subsector = bucket
        return IndustryClassification(
            sector=sector,
            subsector=subsector,
            source=IndustrySource.DIRECT_CATEGORY,
            confidence=DIRECT_CATEGORY_CONFIDENCE,
            evidence=(f"category:{category}",),
            normalized_categories=normalized_categories,
        )
    return None


def _classify_from_skills(
    skills: Iterable[str],
    *,
    config: IndustryTaxonomyConfig,
) -> Optional[IndustryClassification]:
    normalized_skills = [_slugify(skill) for skill in skills if skill]
    matches = [bucket for skill in normalized_skills if (bucket := _SKILL_TO_BUCKET.get(skill))]
    if len(matches) < config.skill_affinity_min_matches:
        return None

    counts = Counter(matches)
    (sector, subsector), count = counts.most_common(1)[0]
    return IndustryClassification(
        sector=sector,
        subsector=subsector,
        source=IndustrySource.SKILL_AFFINITY,
        confidence=count / len(matches),
        evidence=tuple(
            f"skill:{skill}" for skill in normalized_skills if _SKILL_TO_BUCKET.get(skill) == (sector, subsector)
        ),
    )
