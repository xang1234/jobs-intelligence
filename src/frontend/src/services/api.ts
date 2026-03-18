import axios from 'axios'
import type {
  CareerDeltaAnalysisRequest,
  CareerDeltaAnalysisResponse,
  CareerDeltaScenarioDetail,
  CompanySimilarity,
  CompanySimilarityRequest,
  CompanyTrendResponse,
  ErrorResponse,
  HealthResponse,
  OverviewResponse,
  PerformanceStats,
  PopularQuery,
  ProfileMatchRequest,
  ProfileMatchResponse,
  RelatedSkillsResponse,
  RoleTrendRequest,
  RoleTrendResponse,
  SearchRequest,
  SearchResponse,
  SimilarBatchRequest,
  SimilarBatchResponse,
  SimilarJobsRequest,
  SkillCloudResponse,
  SkillSearchRequest,
  SkillTrendRequest,
  SkillTrendSeries,
  StatsResponse,
} from '@/types/api'

const client = axios.create({
  baseURL: '/',
  headers: { 'Content-Type': 'application/json' },
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.data?.error) {
      const apiError = error.response.data as ErrorResponse
      return Promise.reject(new ApiError(
        apiError.error.message,
        apiError.error.code,
        error.response.status,
      ))
    }
    return Promise.reject(error)
  },
)

export class ApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(message: string, code: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
  }
}

export async function searchJobs(req: SearchRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/search', req)
  return data
}

export async function findSimilarJobs(req: SimilarJobsRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/similar', req)
  return data
}

export async function findSimilarJobsBatch(req: SimilarBatchRequest): Promise<SimilarBatchResponse> {
  const { data } = await client.post<SimilarBatchResponse>('/api/similar/batch', req)
  return data
}

export async function searchBySkill(req: SkillSearchRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/search/skills', req)
  return data
}

export async function getSkillCloud(minJobs = 10, limit = 100): Promise<SkillCloudResponse> {
  const { data } = await client.get<SkillCloudResponse>('/api/skills/cloud', {
    params: { min_jobs: minJobs, limit },
  })
  return data
}

export async function getRelatedSkills(skill: string, k = 10): Promise<RelatedSkillsResponse> {
  const { data } = await client.get<RelatedSkillsResponse>(
    `/api/skills/related/${encodeURIComponent(skill)}`,
    { params: { k } },
  )
  return data
}

export async function findSimilarCompanies(req: CompanySimilarityRequest): Promise<CompanySimilarity[]> {
  const { data } = await client.post<CompanySimilarity[]>('/api/companies/similar', req)
  return data
}

export async function getCompanyTrend(companyName: string, months = 12): Promise<CompanyTrendResponse> {
  const { data } = await client.get<CompanyTrendResponse>(
    `/api/trends/companies/${encodeURIComponent(companyName)}`,
    { params: { months } },
  )
  return data
}

export async function getSkillTrends(req: SkillTrendRequest): Promise<SkillTrendSeries[]> {
  const { data } = await client.post<SkillTrendSeries[]>('/api/trends/skills', req)
  return data
}

export async function getRoleTrend(req: RoleTrendRequest): Promise<RoleTrendResponse> {
  const { data } = await client.post<RoleTrendResponse>('/api/trends/roles', req)
  return data
}

export async function matchProfile(req: ProfileMatchRequest): Promise<ProfileMatchResponse> {
  const { data } = await client.post<ProfileMatchResponse>('/api/match/profile', req)
  return data
}

export async function analyzeCareerDelta(req: CareerDeltaAnalysisRequest): Promise<CareerDeltaAnalysisResponse> {
  const { data } = await client.post<CareerDeltaAnalysisResponse>('/api/career-delta', req)
  return data
}

export async function getCareerDeltaScenarioDetail(scenarioId: string): Promise<CareerDeltaScenarioDetail> {
  const { data } = await client.get<CareerDeltaScenarioDetail>(
    `/api/career-delta/${encodeURIComponent(scenarioId)}/detail`,
  )
  return data
}

export async function getOverview(months = 12): Promise<OverviewResponse> {
  const { data } = await client.get<OverviewResponse>('/api/overview', {
    params: { months },
  })
  return data
}

export async function getStats(): Promise<StatsResponse> {
  const { data } = await client.get<StatsResponse>('/api/stats')
  return data
}

export async function getPopularQueries(days = 7, limit = 20): Promise<PopularQuery[]> {
  const { data } = await client.get<PopularQuery[]>('/api/analytics/popular', {
    params: { days, limit },
  })
  return data
}

export async function getPerformanceStats(days = 7): Promise<PerformanceStats> {
  const { data } = await client.get<PerformanceStats>('/api/analytics/performance', {
    params: { days },
  })
  return {
    p50_ms: (data as unknown as { p50?: number }).p50 ?? (data as PerformanceStats).p50_ms,
    p90_ms: (data as unknown as { p90?: number }).p90 ?? (data as PerformanceStats).p90_ms,
    p95_ms: (data as unknown as { p95?: number }).p95 ?? (data as PerformanceStats).p95_ms,
    p99_ms: (data as unknown as { p99?: number }).p99 ?? (data as PerformanceStats).p99_ms,
    total_queries: (data as unknown as { count?: number }).count ?? (data as PerformanceStats).total_queries,
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health')
  return data
}
