type ClassValue = string | false | null | undefined | 0 | 0n

export function cn(...classes: Array<ClassValue | unknown>): string {
  return classes.filter((c): c is string => typeof c === 'string' && c.length > 0).join(' ')
}
