import path from 'node:path'
import { fileURLToPath } from 'node:url'

import playwright from '/Users/admin/.npm/_npx/9833c18b2d85bc59/node_modules/playwright/index.js'

const { chromium } = playwright
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const baseUrl = 'http://127.0.0.1:4173'

const months = [
  '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12',
  '2026-01', '2026-02', '2026-03',
]

function trendSeries(name, counts, salaryBase) {
  return {
    skill: name,
    series: months.map((month, index) => ({
      month,
      job_count: counts[index],
      market_share: Number((counts[index] / 40).toFixed(1)),
      median_salary_annual: salaryBase + index * 1800,
      momentum: index < 3 ? 0 : Number((((counts[index] - ((counts[index - 1] + counts[index - 2] + counts[index - 3]) / 3)) / (((counts[index - 1] + counts[index - 2] + counts[index - 3]) / 3) || 1)) * 100).toFixed(1)),
    })),
    latest: null,
  }
}

const skillTrendPayload = [
  trendSeries('Python', [118, 122, 125, 131, 137, 146, 152, 157, 166, 174, 183, 196, 205, 214, 228], 132000),
  trendSeries('SQL', [140, 142, 145, 149, 150, 154, 156, 160, 164, 168, 171, 176, 182, 188, 194], 126000),
  trendSeries('Machine Learning', [62, 64, 67, 72, 78, 84, 91, 98, 108, 119, 130, 144, 156, 172, 189], 148000),
].map((series) => ({ ...series, latest: series.series.at(-1) }))

const roleTrendPayload = {
  query: 'data scientist',
  series: months.map((month, index) => ({
    month,
    job_count: [74, 76, 79, 81, 84, 88, 92, 97, 103, 108, 116, 121, 126, 132, 139][index],
    market_share: [2.1, 2.1, 2.2, 2.3, 2.4, 2.5, 2.5, 2.6, 2.8, 2.9, 3.0, 3.1, 3.1, 3.2, 3.4][index],
    median_salary_annual: 136000 + index * 2200,
    momentum: [0, 0, 0, 4.8, 5.7, 7.3, 8.5, 9.1, 10.8, 11.6, 12.2, 12.9, 13.5, 14.1, 15.2][index],
  })),
  latest: null,
}
roleTrendPayload.latest = roleTrendPayload.series.at(-1)

const companyTrendPayload = {
  company_name: 'DBS BANK LTD.',
  series: months.map((month, index) => ({
    month,
    job_count: [18, 19, 20, 22, 24, 26, 27, 28, 31, 34, 36, 38, 41, 43, 47][index],
    market_share: [0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.4][index],
    median_salary_annual: 142000 + index * 2400,
    momentum: [0, 0, 0, 8.4, 10.9, 14.7, 12.4, 10.1, 13.8, 16.2, 17.5, 15.4, 14.1, 12.7, 13.2][index],
  })),
  top_skills_by_month: [
    { month: '2025-12', skills: [{ skill: 'Python', job_count: 18, cluster_id: 1 }, { skill: 'Risk', job_count: 14, cluster_id: 2 }, { skill: 'SQL', job_count: 13, cluster_id: 1 }, { skill: 'MLOps', job_count: 10, cluster_id: 4 }, { skill: 'Prompt Engineering', job_count: 8, cluster_id: 5 }] },
    { month: '2026-01', skills: [{ skill: 'Python', job_count: 21, cluster_id: 1 }, { skill: 'Fraud Analytics', job_count: 15, cluster_id: 2 }, { skill: 'SQL', job_count: 15, cluster_id: 1 }, { skill: 'LLM Ops', job_count: 11, cluster_id: 4 }, { skill: 'Airflow', job_count: 9, cluster_id: 3 }] },
    { month: '2026-02', skills: [{ skill: 'Python', job_count: 23, cluster_id: 1 }, { skill: 'Machine Learning', job_count: 19, cluster_id: 4 }, { skill: 'SQL', job_count: 18, cluster_id: 1 }, { skill: 'Risk', job_count: 16, cluster_id: 2 }, { skill: 'AWS', job_count: 11, cluster_id: 6 }] },
    { month: '2026-03', skills: [{ skill: 'Python', job_count: 27, cluster_id: 1 }, { skill: 'Machine Learning', job_count: 22, cluster_id: 4 }, { skill: 'Prompt Engineering', job_count: 18, cluster_id: 5 }, { skill: 'SQL', job_count: 17, cluster_id: 1 }, { skill: 'Feature Store', job_count: 11, cluster_id: 6 }] },
  ],
  similar_companies: [
    { company_name: 'UOB', similarity_score: 0.92, job_count: 112, avg_salary: 151000, top_skills: ['Python', 'Risk', 'ML'] },
    { company_name: 'Standard Chartered', similarity_score: 0.88, job_count: 96, avg_salary: 149000, top_skills: ['Python', 'Fraud Analytics', 'SQL'] },
    { company_name: 'OCBC', similarity_score: 0.84, job_count: 87, avg_salary: 146000, top_skills: ['Risk', 'Analytics', 'AWS'] },
  ],
}

const overviewPayload = {
  headline_metrics: {
    total_jobs: 18472,
    current_month_jobs: 1643,
    unique_companies: 1298,
    unique_skills: 862,
    avg_salary_annual: 138400,
  },
  rising_skills: [
    { name: 'Prompt Engineering', job_count: 196, momentum: 38.6, median_salary_annual: 171000 },
    { name: 'Machine Learning', job_count: 189, momentum: 27.8, median_salary_annual: 164000 },
    { name: 'MLOps', job_count: 142, momentum: 23.4, median_salary_annual: 168000 },
    { name: 'Python', job_count: 228, momentum: 17.6, median_salary_annual: 154000 },
  ],
  rising_companies: [
    { name: 'DBS BANK LTD.', job_count: 47, momentum: 13.2, median_salary_annual: 176000 },
    { name: 'ByteDance', job_count: 35, momentum: 28.4, median_salary_annual: 183000 },
    { name: 'GovTech', job_count: 31, momentum: 18.9, median_salary_annual: 162000 },
    { name: 'Sea Limited', job_count: 26, momentum: 16.1, median_salary_annual: 169000 },
  ],
  salary_movement: {
    current_median_salary_annual: 144000,
    previous_median_salary_annual: 139000,
    change_pct: 3.6,
  },
  market_insights: [
    { label: 'AI and data roles', value: 2418, delta: 12.8 },
    { label: 'Companies hiring for ML', value: 312, delta: 9.7 },
    { label: 'Jobs with salary disclosed', value: 10942, delta: 4.2 },
    { label: 'Senior openings', value: 1786, delta: 7.9 },
  ],
}

const statsPayload = {
  total_jobs: 18472,
  jobs_with_embeddings: 17896,
  embedding_coverage_pct: 96.9,
  unique_skills: 862,
  unique_companies: 1298,
  index_size_mb: 183.4,
  model_version: 'all-MiniLM-L6-v2',
}

const popularPayload = [
  { query: 'data scientist', count: 186 },
  { query: 'machine learning engineer', count: 143 },
  { query: 'python developer', count: 131 },
  { query: 'ai product manager', count: 92 },
  { query: 'llm engineer', count: 88 },
]

const performancePayload = {
  p50: 91,
  p90: 188,
  p95: 244,
  p99: 388,
  count: 842,
}

const searchPayload = {
  results: [
    {
      uuid: 'job-1',
      title: 'Senior Machine Learning Engineer',
      company_name: 'Sea Limited',
      description: 'Lead recommender systems, feature engineering, experimentation, and model deployment across consumer products.',
      salary_min: 138000,
      salary_max: 186000,
      employment_type: 'Full Time',
      seniority: 'Senior',
      skills: 'Python, Machine Learning, SQL, Airflow, Experimentation, AWS',
      location: 'Singapore',
      posted_date: '2026-03-10',
      job_url: 'https://example.com/job-1',
      similarity_score: 0.91,
      semantic_score: 0.89,
      bm25_score: 0.83,
      freshness_score: 0.77,
      matched_skills: ['Python', 'Machine Learning', 'SQL', 'Experimentation'],
      missing_skills: ['Feature Store', 'Kubernetes'],
      explanations: {
        semantic_score: 0.89,
        bm25_score: 0.83,
        freshness_score: 0.77,
        matched_skills: ['Python', 'Machine Learning', 'SQL', 'Experimentation'],
        missing_skills: ['Feature Store', 'Kubernetes'],
        query_terms: ['machine learning engineer', 'ml engineer', 'recommender systems', 'python'],
        skill_overlap_score: 0.8,
        seniority_fit: 0.9,
        salary_fit: 0.86,
        overall_fit: 0.91,
      },
    },
    {
      uuid: 'job-2',
      title: 'Applied AI Engineer',
      company_name: 'ByteDance',
      description: 'Build retrieval, ranking, and LLM evaluation pipelines for growth and monetization products.',
      salary_min: 142000,
      salary_max: 198000,
      employment_type: 'Full Time',
      seniority: 'Senior',
      skills: 'Python, NLP, LLM Ops, SQL, Retrieval, Prompt Engineering',
      location: 'Singapore',
      posted_date: '2026-03-08',
      job_url: 'https://example.com/job-2',
      similarity_score: 0.87,
      semantic_score: 0.84,
      bm25_score: 0.79,
      freshness_score: 0.75,
      matched_skills: ['Python', 'NLP', 'SQL'],
      missing_skills: ['Airflow', 'Experimentation'],
      explanations: {
        semantic_score: 0.84,
        bm25_score: 0.79,
        freshness_score: 0.75,
        matched_skills: ['Python', 'NLP', 'SQL'],
        missing_skills: ['Airflow', 'Experimentation'],
        query_terms: ['machine learning engineer', 'applied ai', 'nlp', 'retrieval'],
        skill_overlap_score: 0.72,
        seniority_fit: 0.88,
        salary_fit: 0.83,
        overall_fit: 0.87,
      },
    },
  ],
  total_candidates: 342,
  search_time_ms: 184,
  query_expansion: ['machine learning engineer', 'ml engineer', 'ai engineer', 'python', 'nlp'],
  degraded: false,
  cache_hit: false,
}

const matchPayload = {
  results: [
    {
      uuid: 'match-1',
      title: 'Lead Data Scientist, Personalization',
      company_name: 'Grab',
      description: 'Own ranking models, experimentation, and product analytics for high-scale marketplace experiences.',
      salary_min: 152000,
      salary_max: 202000,
      employment_type: 'Full Time',
      seniority: 'Lead',
      skills: 'Python, SQL, Machine Learning, Experimentation, Stakeholder Management, Dashboards',
      location: 'Singapore',
      posted_date: '2026-03-09',
      job_url: 'https://example.com/match-1',
      similarity_score: 0.93,
      semantic_score: 0.9,
      bm25_score: 0.81,
      freshness_score: 0.73,
      matched_skills: ['Python', 'SQL', 'Machine Learning', 'Experimentation', 'Stakeholder Management'],
      missing_skills: ['MLOps'],
      explanations: {
        semantic_score: 0.9,
        bm25_score: 0.81,
        freshness_score: 0.73,
        matched_skills: ['Python', 'SQL', 'Machine Learning', 'Experimentation', 'Stakeholder Management'],
        missing_skills: ['MLOps'],
        query_terms: ['data scientist', 'machine learning engineer', 'analytics'],
        skill_overlap_score: 0.86,
        seniority_fit: 0.94,
        salary_fit: 0.88,
        overall_fit: 0.93,
      },
    },
    {
      uuid: 'match-2',
      title: 'Senior Applied ML Engineer',
      company_name: 'GovTech',
      description: 'Build production ML systems for public sector services with strong stakeholder and delivery ownership.',
      salary_min: 146000,
      salary_max: 188000,
      employment_type: 'Full Time',
      seniority: 'Senior',
      skills: 'Python, SQL, Machine Learning, Airflow, Stakeholder Management, Analytics',
      location: 'Singapore',
      posted_date: '2026-03-11',
      job_url: 'https://example.com/match-2',
      similarity_score: 0.88,
      semantic_score: 0.85,
      bm25_score: 0.76,
      freshness_score: 0.8,
      matched_skills: ['Python', 'SQL', 'Machine Learning', 'Stakeholder Management'],
      missing_skills: ['Experimentation'],
      explanations: {
        semantic_score: 0.85,
        bm25_score: 0.76,
        freshness_score: 0.8,
        matched_skills: ['Python', 'SQL', 'Machine Learning', 'Stakeholder Management'],
        missing_skills: ['Experimentation'],
        query_terms: ['data scientist', 'applied ml', 'analytics'],
        skill_overlap_score: 0.78,
        seniority_fit: 0.88,
        salary_fit: 0.84,
        overall_fit: 0.88,
      },
    },
  ],
  extracted_skills: ['Python', 'SQL', 'Machine Learning', 'Experimentation', 'Stakeholder Management', 'Dashboards'],
  total_candidates: 284,
  search_time_ms: 231,
  degraded: false,
}

const skillCloudPayload = {
  items: [
    { skill: 'Python', job_count: 228, cluster_id: 1 },
    { skill: 'SQL', job_count: 194, cluster_id: 1 },
    { skill: 'Machine Learning', job_count: 189, cluster_id: 4 },
    { skill: 'Prompt Engineering', job_count: 196, cluster_id: 5 },
    { skill: 'MLOps', job_count: 142, cluster_id: 4 },
    { skill: 'Airflow', job_count: 137, cluster_id: 3 },
    { skill: 'AWS', job_count: 176, cluster_id: 6 },
    { skill: 'NLP', job_count: 118, cluster_id: 5 },
    { skill: 'Experimentation', job_count: 126, cluster_id: 2 },
  ],
  total_unique_skills: 862,
}

const relatedSkillsPayload = {
  skill: 'Machine Learning',
  related: [
    { skill: 'MLOps', similarity: 0.92, same_cluster: true },
    { skill: 'Deep Learning', similarity: 0.91, same_cluster: true },
    { skill: 'LLM Ops', similarity: 0.88, same_cluster: true },
    { skill: 'Feature Store', similarity: 0.84, same_cluster: false },
  ],
}

const companySimilarityPayload = [
  { company_name: 'Grab', similarity_score: 0.9, job_count: 94, avg_salary: 168000, top_skills: ['Python', 'Experimentation', 'ML'] },
  { company_name: 'Sea Limited', similarity_score: 0.87, job_count: 76, avg_salary: 173000, top_skills: ['Python', 'Recommenders', 'SQL'] },
]

function respond(route, payload) {
  return route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

async function installApiMocks(page) {
  await page.route('**/health', async (route) => respond(route, { status: 'ok', index_loaded: true, degraded: false }))

  await page.route('**/api/overview**', async (route) => respond(route, overviewPayload))
  await page.route('**/api/stats**', async (route) => respond(route, statsPayload))
  await page.route('**/api/analytics/popular**', async (route) => respond(route, popularPayload))
  await page.route('**/api/analytics/performance**', async (route) => respond(route, performancePayload))
  await page.route('**/api/trends/skills**', async (route) => respond(route, skillTrendPayload))
  await page.route('**/api/trends/roles**', async (route) => respond(route, roleTrendPayload))
  await page.route('**/api/trends/companies/**', async (route) => respond(route, companyTrendPayload))
  await page.route('**/api/match/profile**', async (route) => respond(route, matchPayload))
  await page.route('**/api/search', async (route) => respond(route, searchPayload))
  await page.route('**/api/skills/cloud**', async (route) => respond(route, skillCloudPayload))
  await page.route('**/api/skills/related/**', async (route) => respond(route, relatedSkillsPayload))
  await page.route('**/api/companies/similar**', async (route) => respond(route, companySimilarityPayload))
}

async function save(page, name, options = {}) {
  await page.screenshot({
    path: path.join(__dirname, name),
    type: 'png',
    ...options,
  })
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  })
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1200 },
    deviceScaleFactor: 1.5,
  })

  await installApiMocks(page)

  await page.goto(`${baseUrl}/`, { waitUntil: 'domcontentloaded' })
  await page.getByText('Prompt Engineering').waitFor({ timeout: 15000 })
  await save(page, 'overview-dashboard.png')

  await page.goto(`${baseUrl}/trends`, { waitUntil: 'domcontentloaded' })
  await page.getByText('Standard Chartered').waitFor({ timeout: 15000 })
  await save(page, 'trends-explorer.png', { fullPage: true })

  await page.goto(`${baseUrl}/match-lab`, { waitUntil: 'domcontentloaded' })
  await page.getByRole('button', { name: 'Run profile match' }).click()
  await page.getByText('Lead Data Scientist, Personalization').waitFor({ timeout: 15000 })
  await save(page, 'match-lab-results.png')

  await page.goto(`${baseUrl}/search`, { waitUntil: 'domcontentloaded' })
  await page.getByPlaceholder('Search jobs (e.g. data scientist, python developer)').fill('machine learning engineer')
  await page.getByRole('button', { name: 'Search' }).click()
  await page.getByText('Senior Machine Learning Engineer').waitFor({ timeout: 15000 })
  await save(page, 'search-similarity.png')

  await browser.close()
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
