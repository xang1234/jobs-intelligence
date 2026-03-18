import type {
  CareerDeltaAnalysisRequest,
  MatchLabCareerDeltaOptions,
  MatchLabSharedInputs,
  ProfileMatchRequest,
} from '@/types/api'

function splitCommaSeparated(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

function parseOptionalNumber(value: string): number | null {
  if (!value.trim()) {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function buildProfileMatchRequest(
  inputs: MatchLabSharedInputs,
  limit = 12,
): ProfileMatchRequest {
  return {
    profile_text: inputs.profileText,
    target_titles: splitCommaSeparated(inputs.targetTitles),
    salary_expectation_annual: parseOptionalNumber(inputs.salaryExpectation),
    employment_type: inputs.employmentType || null,
    region: inputs.region || null,
    limit,
  }
}

export function buildCareerDeltaAnalysisRequest(
  inputs: MatchLabSharedInputs,
  options: MatchLabCareerDeltaOptions = {},
): CareerDeltaAnalysisRequest {
  return {
    profile_text: inputs.profileText,
    current_title: options.currentTitle ?? null,
    target_titles: splitCommaSeparated(inputs.targetTitles),
    current_categories: options.currentCategories ?? [],
    current_skills: options.currentSkills ?? [],
    current_company: options.currentCompany ?? null,
    location: options.location ?? (inputs.region || null),
    target_salary_min: options.targetSalaryMin ?? parseOptionalNumber(inputs.salaryExpectation),
    max_scenarios: options.maxScenarios ?? 6,
    include_filtered: options.includeFiltered ?? true,
    delta_types: options.deltaTypes ?? [],
  }
}
