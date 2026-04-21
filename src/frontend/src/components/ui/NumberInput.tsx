import { forwardRef, useCallback, type ChangeEvent } from 'react'
import { Input, type InputProps } from './Input'

export interface NumberInputProps
  extends Omit<InputProps, 'type' | 'onChange' | 'value' | 'defaultValue'> {
  value?: number | null
  onValueChange?: (value: number | null) => void
  min?: number
  max?: number
  step?: number
}

export const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(
  function NumberInput({ value, onValueChange, onBlur, ...rest }, ref) {
    const handleChange = useCallback(
      (event: ChangeEvent<HTMLInputElement>) => {
        const raw = event.target.value
        if (raw === '' || raw == null) {
          onValueChange?.(null)
          return
        }
        const parsed = Number(raw)
        if (Number.isNaN(parsed)) return
        onValueChange?.(parsed)
      },
      [onValueChange],
    )

    return (
      <Input
        ref={ref}
        type="number"
        inputMode="numeric"
        value={value ?? ''}
        onChange={handleChange}
        onBlur={onBlur}
        {...rest}
      />
    )
  },
)
