import assert from 'node:assert/strict'
import { test } from 'node:test'

import { resolveApiBaseUrl } from '../src/services/apiBaseUrl.ts'

test('uses same-origin API routes when VITE_API_BASE_URL is not configured', () => {
  assert.equal(resolveApiBaseUrl({}), '/')
  assert.equal(resolveApiBaseUrl({ VITE_API_BASE_URL: '   ' }), '/')
})

test('uses configured absolute API base URL for hosted frontend deployments', () => {
  assert.equal(
    resolveApiBaseUrl({ VITE_API_BASE_URL: 'https://xang1234-jobs-intelligence-api.hf.space' }),
    'https://xang1234-jobs-intelligence-api.hf.space',
  )
})

test('normalizes configured API base URL whitespace and trailing slash', () => {
  assert.equal(
    resolveApiBaseUrl({ VITE_API_BASE_URL: ' https://xang1234-jobs-intelligence-api.hf.space/ ' }),
    'https://xang1234-jobs-intelligence-api.hf.space',
  )
})
