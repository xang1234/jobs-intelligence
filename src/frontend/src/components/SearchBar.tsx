import { useState, forwardRef, useImperativeHandle, useRef } from 'react'
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { Button, IconButton, Input } from '@/components/ui'

interface SearchBarProps {
  onSearch: (query: string) => void
  isLoading: boolean
  defaultValue?: string
  placeholder?: string
  submitLabel?: string
}

export interface SearchBarHandle {
  focus: () => void
  setValue: (value: string) => void
}

const SearchBar = forwardRef<SearchBarHandle, SearchBarProps>(function SearchBar(
  {
    onSearch,
    isLoading,
    defaultValue = '',
    placeholder = 'Search jobs (e.g. data scientist, python developer)',
    submitLabel = 'Search',
  },
  ref,
) {
  const [input, setInput] = useState(defaultValue)
  const inputRef = useRef<HTMLInputElement>(null)

  useImperativeHandle(
    ref,
    () => ({
      focus: () => inputRef.current?.focus(),
      setValue: (value: string) => setInput(value),
    }),
    [],
  )

  function handleSubmit() {
    const trimmed = input.trim()
    if (trimmed) onSearch(trimmed)
  }

  return (
    <div className="flex gap-2">
      <div className="flex-1">
        <Input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSubmit()
          }}
          placeholder={placeholder}
          leftIcon={<MagnifyingGlassIcon />}
          rightIcon={
            input ? (
              <IconButton
                aria-label="Clear search"
                icon={<XMarkIcon />}
                size="sm"
                variant="ghost"
                onClick={() => {
                  setInput('')
                  inputRef.current?.focus()
                }}
              />
            ) : undefined
          }
          aria-label="Search query"
        />
      </div>
      <Button onClick={handleSubmit} disabled={!input.trim()} loading={isLoading}>
        {submitLabel}
      </Button>
    </div>
  )
})

export default SearchBar
