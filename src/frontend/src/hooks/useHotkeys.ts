import { useEffect } from 'react'

type Combo = string // e.g. "mod+k", "/", "Escape"

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  if (target.isContentEditable) return true
  return false
}

function matches(event: KeyboardEvent, combo: Combo): boolean {
  const parts = combo.toLowerCase().split('+').map((s) => s.trim())
  const key = parts[parts.length - 1]
  const mods = new Set(parts.slice(0, -1))

  const needsMod = mods.has('mod') || mods.has('cmd') || mods.has('ctrl')
  const needsShift = mods.has('shift')
  const needsAlt = mods.has('alt') || mods.has('option')

  const gotMod = event.metaKey || event.ctrlKey
  if (needsMod !== gotMod) return false
  if (needsShift !== event.shiftKey) return false
  if (needsAlt !== event.altKey) return false

  return event.key.toLowerCase() === key
}

export function useHotkeys(
  map: Record<Combo, (e: KeyboardEvent) => void>,
  { allowInInputs = false }: { allowInInputs?: boolean } = {},
) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!allowInInputs && isTypingTarget(event.target)) {
        // Allow Escape & mod+k even when typing
        if (event.key !== 'Escape' && !((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
          return
        }
      }
      for (const [combo, fn] of Object.entries(map)) {
        if (matches(event, combo)) {
          fn(event)
          return
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [map, allowInInputs])
}
