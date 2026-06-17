import assert from 'node:assert/strict'
import { test } from 'node:test'

import { snapshotFirst } from '../src/services/snapshots.ts'

test('snapshotFirst returns the snapshot when fetch succeeds', async () => {
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ ok: 'snapshot' }), { status: 200 })) as typeof fetch
  let apiCalled = false
  const fn = snapshotFirst<{ ok: string }>('overview', async () => {
    apiCalled = true
    return { ok: 'api' }
  })
  const result = await fn()
  assert.equal(result.ok, 'snapshot')
  assert.equal(apiCalled, false)
})

test('snapshotFirst falls back to the API on 404', async () => {
  globalThis.fetch = (async () => new Response('not found', { status: 404 })) as typeof fetch
  const fn = snapshotFirst<{ ok: string }>('overview', async () => ({ ok: 'api' }))
  const result = await fn()
  assert.equal(result.ok, 'api')
})

test('snapshotFirst falls back to the API when fetch throws', async () => {
  globalThis.fetch = (async () => {
    throw new Error('network down')
  }) as typeof fetch
  const fn = snapshotFirst<{ ok: string }>('overview', async () => ({ ok: 'api' }))
  const result = await fn()
  assert.equal(result.ok, 'api')
})
