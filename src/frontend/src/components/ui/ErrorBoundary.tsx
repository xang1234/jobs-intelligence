import { Component, type ErrorInfo, type ReactNode } from 'react'
import { ErrorState } from './ErrorState'

interface Props {
  children: ReactNode
  fallback?: (error: Error, reset: () => void) => ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    if (import.meta.env.DEV) {
      console.error('ErrorBoundary caught error:', error, info)
    }
  }

  reset = () => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.fallback) return this.props.fallback(error, this.reset)
    return (
      <div className="mx-auto max-w-xl px-6 py-12">
        <ErrorState
          title="Unexpected error"
          description={
            import.meta.env.DEV
              ? error.message || 'An unexpected error occurred.'
              : 'An unexpected error occurred.'
          }
          retryLabel="Reload"
          onRetry={() => {
            this.reset()
            window.location.reload()
          }}
        />
      </div>
    )
  }
}
