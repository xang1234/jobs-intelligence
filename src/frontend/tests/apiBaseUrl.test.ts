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

test('preserves a single slash for same-origin API routes', () => {
  assert.equal(resolveApiBaseUrl({ VITE_API_BASE_URL: '/' }), '/')
})

test('falls back to same-origin routes when configured API base URL is only slashes', () => {
  assert.equal(resolveApiBaseUrl({ VITE_API_BASE_URL: '//' }), '/')
})

test('normalizes multiple trailing slashes from absolute API base URL', () => {
  assert.equal(resolveApiBaseUrl({ VITE_API_BASE_URL: 'https://example.com///' }), 'https://example.com')
})
