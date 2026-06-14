import assert from 'node:assert/strict'
import { after, before, test } from 'node:test'
import fs from 'node:fs/promises'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const frontendDir = path.resolve(__dirname, '..')
const serverUrl = 'http://127.0.0.1:4175'

let serverProcess
let chromium

const trendPoint = (month, jobCount, overrides = {}) => ({
  month,
  job_count: jobCount,
  market_share: jobCount ? 2.58 : 0,
  median_salary_annual: jobCount ? 51000 : null,
  momentum: jobCount ? 100 : 0,
  momentum_status: jobCount ? 'new' : 'insufficient_baseline',
  momentum_label: jobCount ? 'New signal' : 'No prior baseline',
  ...overrides,
})

const trendSeries = [
  trendPoint('2026-01', 0),
  trendPoint('2026-02', 0),
  trendPoint('2026-03', 70),
]

const overviewResponse = {
  headline_metrics: {
    total_jobs: 2711,
    current_month_jobs: 2711,
    unique_companies: 900,
    unique_skills: 1200,
    avg_salary_annual: 65000,
  },
  rising_skills: [],
  rising_companies: [],
  salary_movement: {
    current_median_salary_annual: 65000,
    previous_median_salary_annual: null,
    change_pct: 0,
  },
  market_insights: [],
}

const companyTrendResponse = {
  company_name: 'RECRUIT EXPERT PTE. LTD.',
  series: trendSeries,
  top_skills_by_month: [
    { month: '2026-01', skills: [] },
    { month: '2026-02', skills: [] },
    {
      month: '2026-03',
      skills: [
        { skill: 'Construction', job_count: 21, cluster_id: null },
        { skill: 'MS Word', job_count: 16, cluster_id: null },
      ],
    },
  ],
  similar_companies: [],
}

const similarCompaniesResponse = [
  {
    company_name: 'BUILD TEAM PTE. LTD.',
    similarity_score: 0.84,
    job_count: 17,
    avg_salary: 62000,
    top_skills: ['Construction'],
  },
]

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
      if (response.ok) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  throw new Error(`Timed out waiting for dev server at ${url}`)
}

function startDevServer() {
  return spawn(
    process.execPath,
    ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4175', '--strictPort'],
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

function apiResponseFor(requestUrl, method) {
  const url = new URL(requestUrl)

  if (url.pathname === '/api/overview') {
    return overviewResponse
  }
  if (url.pathname === '/api/trends/skills') {
    return [
      { skill: 'Customer Service', series: trendSeries, latest: trendSeries.at(-1) },
      { skill: 'Microsoft Excel', series: trendSeries, latest: trendSeries.at(-1) },
      { skill: 'Communication Skills', series: trendSeries, latest: trendSeries.at(-1) },
    ]
  }
  if (url.pathname === '/api/trends/roles') {
    return {
      query: 'customer service',
      series: trendSeries,
      latest: trendSeries.at(-1),
    }
  }
  if (url.pathname.startsWith('/api/trends/companies/')) {
    assert.equal(method, 'GET')
    assert.equal(url.searchParams.get('include_similar'), 'false')
    return companyTrendResponse
  }
  if (url.pathname === '/api/companies/similar') {
    return similarCompaniesResponse
  }
  if (url.pathname === '/api/stats') {
    return {
      total_jobs: 2711,
      jobs_with_embeddings: 2711,
      embedding_coverage_pct: 100,
      unique_skills: 1200,
      unique_companies: 900,
      index_size_mb: 12,
      model_version: 'test',
    }
  }
  if (url.pathname === '/api/analytics/popular') {
    return []
  }
  if (url.pathname === '/api/analytics/performance') {
    return { p50_ms: 10, p90_ms: 20, p95_ms: 25, p99_ms: 30, total_queries: 5 }
  }
  if (url.pathname === '/api/skills/cloud') {
    return { items: [], total_unique_skills: 0 }
  }
  if (url.pathname === '/health') {
    return { status: 'ok', index_loaded: true, degraded: false }
  }

  throw new Error(`Unexpected ${method} ${url.pathname}`)
}

before(async () => {
  ;({ chromium } = await loadPlaywright())
  serverProcess = startDevServer()
  await waitForServer(`${serverUrl}/trends`)
})

after(async () => {
  if (serverProcess && !serverProcess.killed) {
    serverProcess.kill('SIGTERM')
    await new Promise((resolve) => serverProcess.once('exit', resolve))
  }
})

test('loads similar employer profiles only after an explicit request', async () => {
  let similarRequests = 0
  const browser = await launchBrowser()
  const context = await browser.newContext()
  const page = await context.newPage()

  try {
    await page.route('**/{api,health}/**', async (route) => {
      const request = route.request()
      const url = request.url()
      if (new URL(url).pathname === '/api/companies/similar') {
        similarRequests += 1
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(apiResponseFor(url, request.method())),
      })
    })

    await page.goto(`${serverUrl}/trends`)
    await page.getByText('RECRUIT EXPERT PTE. LTD.').waitFor()
    await page.waitForTimeout(500)

    assert.equal(similarRequests, 0)

    await page.getByRole('button', { name: 'Load similar profiles' }).click()
    await page.getByText('BUILD TEAM PTE. LTD.').waitFor()
    assert.equal(similarRequests, 1)
  } finally {
    await context.close()
    await browser.close()
  }
})
