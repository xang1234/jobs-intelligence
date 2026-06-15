import assert from 'node:assert/strict'
import { afterEach, test } from 'node:test'

import { scheduleIdleQueue, scheduleIdleTask } from '../src/services/idle.ts'

const originalWindow = globalThis.window
const originalNavigator = globalThis.navigator

type ScheduledCallback = () => void

function installIdleGlobals(connection: { saveData?: boolean; effectiveType?: string } = {}) {
  const scheduled: ScheduledCallback[] = []
  const fakeWindow = {
    setTimeout: (callback: ScheduledCallback) => {
      scheduled.push(callback)
      return scheduled.length
    },
    clearTimeout: () => {},
    requestIdleCallback: (callback: ScheduledCallback) => {
      scheduled.push(callback)
      return scheduled.length
    },
    cancelIdleCallback: () => {},
  }

  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: fakeWindow,
  })
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { connection },
  })

  return scheduled
}

afterEach(() => {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: originalWindow,
  })
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: originalNavigator,
  })
})

test('runs visible idle tasks even on constrained connections', () => {
  const scheduled = installIdleGlobals({ saveData: true })
  let ran = false

  scheduleIdleTask(() => {
    ran = true
  })

  scheduled.shift()?.()
  scheduled.shift()?.()

  assert.equal(ran, true)
})

test('skips speculative idle queues on constrained connections by default', () => {
  const scheduled = installIdleGlobals({ effectiveType: '2g' })
  let ran = false

  scheduleIdleQueue([
    () => {
      ran = true
    },
  ])

  assert.equal(scheduled.length, 0)
  assert.equal(ran, false)
})
