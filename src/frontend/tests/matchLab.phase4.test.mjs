import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import fs from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const frontendDir = path.resolve(__dirname, '..')
const serverUrl = 'http://127.0.0.1:4174'

let serverProcess
let chromium

const baseCareerDeltaResponse = {
  baseline: {
    position: 'competitive',
    reachable_jobs: 24,
    total_candidates: 180,
    fit_median: 0.72,
    fit_p90: 0.88,
    skill_coverage: 0.66,
    salary_band: {
      min_annual: 120000,
      median_annual: 145000,
      max_annual: 178000,
    },
    extracted_skills: ['Python', 'SQL', 'Machine Learning'],
    top_skill_gaps: [
      { name: 'MLOps', job_count: 12, share_pct: 50 },
      { name: 'Kubernetes', job_count: 8, share_pct: 33 },
    ],
    top_industries: [
      { name: 'technology/data_and_ai', job_count: 11, share_pct: 46 },
    ],
    top_companies: [
      { name: 'Sea', job_count: 5, share_pct: 21 },
    ],
    notes: ['baseline note'],
    thin_market: false,
    degraded: false,
  },
  scenarios: [
    {
      scenario_id: 'scenario-1',
      scenario_type: 'title_pivot',
      title: 'Shift into platform-focused ML roles',
      summary: 'Platform ML roles show better salary and stronger demand than your current baseline.',
      market_position: 'competitive',
      confidence: {
        score: 0.82,
        reasons: ['strong demand cluster'],
        evidence_coverage: 0.74,
        market_sample_size: 18,
      },
      score_breakdown: {
        opportunity: 0.88,
        quality: 0.79,
        salary: 0.61,
        momentum: 0.55,
        diversity: 0.45,
        pivot_cost: 0.3,
        final_score: 0.71,
      },
      change: {
        source_title_family: 'data_scientist',
        target_title_family: 'platform_engineer',
        source_industry: 'technology/data_and_ai',
        target_industry: 'technology/data_and_ai',
        added_skills: ['Kubernetes'],
        removed_skills: [],
        replaced_skills: [],
      },
      signals: [
        {
          skill: 'Kubernetes',
          supporting_jobs: 9,
          supporting_share_pct: 50,
          fit_median: 0.76,
          market_salary_annual_median: 168000,
          market_momentum: 18,
        },
      ],
      target_title: 'Platform Machine Learning Engineer',
      target_sector: 'technology/data_and_ai',
      thin_market: false,
      degraded: false,
      expected_salary_delta_pct: 0.16,
    },
  ],
  filtered_scenarios: [],
  degraded: false,
  thin_market: false,
  analysis_time_ms: 132,
}

const baseDetailResponse = {
  scenario_id: 'scenario-1',
  scenario_type: 'title_pivot',
  title: 'Shift into platform-focused ML roles',
  narrative: 'This move trades some experimentation emphasis for stronger platform and deployment demand.',
  market_position: 'competitive',
  confidence: {
    score: 0.82,
    reasons: ['strong demand cluster'],
    evidence_coverage: 0.74,
    market_sample_size: 18,
  },
  score_breakdown: {
    opportunity: 0.88,
    quality: 0.79,
    salary: 0.61,
    momentum: 0.55,
    diversity: 0.45,
    pivot_cost: 0.3,
    final_score: 0.71,
  },
  summary: baseCareerDeltaResponse.scenarios[0],
  change: baseCareerDeltaResponse.scenarios[0].change,
  signals: baseCareerDeltaResponse.scenarios[0].signals,
  target_title: 'Platform Machine Learning Engineer',
  target_sector: 'technology/data_and_ai',
  evidence: ['9 reachable jobs ask for Kubernetes and platform deployment experience.'],
  missing_skills: ['Kubernetes'],
  search_queries: ['platform machine learning engineer singapore', 'mlops kubernetes singapore'],
  thin_market: false,
  degraded: false,
}

const matchResponse = {
  results: [
    {
      uuid: 'job-1',
      title: 'Platform Machine Learning Engineer',
      company_name: 'Sea',
      description: 'Build model-serving and deployment systems.',
      salary_min: 140000,
      salary_max: 182000,
      employment_type: 'Full Time',
      seniority: 'Senior',
      skills: 'Python, Kubernetes, MLOps',
      location: 'Singapore',
      posted_date: '2026-03-18',
      job_url: 'https://example.com/job-1',
      similarity_score: 0.91,
      semantic_score: 0.88,
      bm25_score: 0.84,
      freshness_score: 0.75,
      matched_skills: ['Python'],
      missing_skills: ['Kubernetes'],
      explanations: {
        semantic_score: 0.88,
        bm25_score: 0.84,
        freshness_score: 0.75,
        matched_skills: ['Python'],
        missing_skills: ['Kubernetes'],
        query_terms: ['platform machine learning engineer'],
        skill_overlap_score: 0.63,
        seniority_fit: 0.81,
        salary_fit: 0.77,
        overall_fit: 0.91,
      },
    },
  ],
  total_candidates: 42,
  search_time_ms: 91,
  query_expansion: null,
  degraded: false,
  cache_hit: false,
}

async function loadPlaywright() {
  try {
    const module = await import('playwright')
    return module.default ?? module
  } catch {}

  const npxRoot = path.join(process.env.HOME ?? '', '.npm', '_npx')
  const entries = await fs.readdir(npxRoot).catch(() => [])
  for (const entry of entries) {
    const candidate = path.join(npxRoot, entry, 'node_modules', 'playwright', 'index.js')
    try {
      await fs.access(candidate)
      const module = await import(pathToFileURL(candidate).href)
      return module.default ?? module
    } catch {}
  }

  throw new Error('Playwright could not be resolved. Run tests with `npx --package playwright node --test ...`.')
}

async function waitForServer(url, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url)
      if (response.ok) {
        return
      }
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for dev server at ${url}`)
}

function startDevServer() {
  return spawn(
    process.execPath,
    ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4174', '--strictPort'],
    {
      cwd: frontendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env },
    },
  )
}

async function launchBrowser() {
  try {
    return await chromium.launch()
  } catch (error) {
    try {
      return await chromium.launch({ channel: 'chrome' })
    } catch {
      throw error
    }
  }
}

async function withPage(setupRoutes, runAssertions) {
  const browser = await launchBrowser()
  const context = await browser.newContext()
  const page = await context.newPage()
  try {
    await setupRoutes(page)
    await page.goto(`${serverUrl}/match-lab`)
    await runAssertions(page)
  } finally {
    await context.close()
    await browser.close()
  }
}

before(async () => {
  ;({ chromium } = await loadPlaywright())
  serverProcess = startDevServer()
  await waitForServer(`${serverUrl}/match-lab`)
})

after(async () => {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill('SIGTERM')
    await new Promise((resolve) => serverProcess.once('exit', resolve))
  }
})

test('renders what-if summary recommendations and trust cues', async () => {
  const analysisResponse = structuredClone(baseCareerDeltaResponse)
  analysisResponse.degraded = true
  analysisResponse.thin_market = true
  analysisResponse.baseline.degraded = true
  analysisResponse.baseline.thin_market = true
  analysisResponse.filtered_scenarios = [
    {
      scenario_id: 'filtered-1',
      scenario_type: 'adjacent_role',
      reason_code: 'budget_exhausted',
      explanation: 'Budget exhausted before ranking lower-signal adjacent roles.',
      confidence: {
        score: 0.41,
        reasons: ['partial pass'],
        evidence_coverage: 0.2,
        market_sample_size: 4,
      },
    },
  ]

  await withPage(
    async (page) => {
      await page.route('**/api/career-delta', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(analysisResponse),
        })
      })
    },
    async (page) => {
      await page.getByRole('button', { name: 'Run What If' }).click()

      await page.getByText('Shift into platform-focused ML roles').waitFor()
      await page.getByText('There is not much reliable market evidence for this profile').waitFor()
      await page.getByText('The engine returned a partial recommendation set').waitFor()
      await page.getByText('Rejected moves the engine considered').waitFor()
      await page.getByText('Platform ML roles show better salary and stronger demand than your current baseline.').waitFor()
    },
  )
})

test('loads scenario detail only when expanded', async () => {
  let detailRequests = 0

  await withPage(
    async (page) => {
      await page.route('**/api/career-delta', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(baseCareerDeltaResponse),
        })
      })
      await page.route('**/api/career-delta/scenario-1/detail', async (route) => {
        detailRequests += 1
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(baseDetailResponse),
        })
      })
    },
    async (page) => {
      await page.getByRole('button', { name: 'Run What If' }).click()
      await page.getByText('Shift into platform-focused ML roles').waitFor()

      assert.equal(detailRequests, 0)

      await page.getByRole('button', { name: 'Inspect detail' }).click()

      await page.getByText('Counterfactual angle').waitFor()
      await page.getByText(baseDetailResponse.narrative).waitFor()
      await page.getByText(baseDetailResponse.evidence[0]).waitFor()
      await page.getByText('Missing skills to validate: Kubernetes').waitFor()
      await page.getByText('platform machine learning engineer singapore').waitFor()
      assert.equal(detailRequests, 1)
    },
  )
})

test('applying a scenario updates visible inputs and reruns profile matching', async () => {
  const matchRequests = []

  await withPage(
    async (page) => {
      await page.route('**/api/career-delta', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(baseCareerDeltaResponse),
        })
      })
      await page.route('**/api/career-delta/scenario-1/detail', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(baseDetailResponse),
        })
      })
      await page.route('**/api/match/profile', async (route) => {
        matchRequests.push(JSON.parse(route.request().postData() ?? '{}'))
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(matchResponse),
        })
      })
    },
    async (page) => {
      const targetTitlesInput = page.getByLabel('Target titles')
      const originalTargetTitles = await targetTitlesInput.inputValue()
      const originalProfileText = await page.getByLabel('Candidate profile or resume text').inputValue()

      await page.getByRole('button', { name: 'Run What If' }).click()
      await page.getByText('Shift into platform-focused ML roles').waitFor()

      await page.getByRole('button', { name: 'Inspect detail' }).click()
      await page.getByText(baseDetailResponse.narrative).waitFor()
      await page.getByRole('button', { name: 'Apply this scenario' }).click()

      await page.getByText('Candidates scanned:').waitFor()
      assert.equal(await targetTitlesInput.inputValue(), 'Platform Machine Learning Engineer')

      assert.equal(matchRequests.length, 1)
      assert.deepEqual(matchRequests[0].target_titles, ['Platform Machine Learning Engineer'])
      assert.equal(matchRequests[0].profile_text, originalProfileText)
      assert.equal(originalTargetTitles, 'Data Scientist, Machine Learning Engineer')
    },
  )
})
