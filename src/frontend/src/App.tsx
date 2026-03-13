import { NavLink, Route, Routes } from 'react-router-dom'
import MatchLabPage from '@/pages/MatchLabPage'
import OverviewPage from '@/pages/OverviewPage'
import SearchPage from '@/pages/SearchPage'
import TrendsPage from '@/pages/TrendsPage'

const NAV_ITEMS = [
  { to: '/', label: 'Overview' },
  { to: '/trends', label: 'Trends Explorer' },
  { to: '/match-lab', label: 'Match Lab' },
  { to: '/search', label: 'Search & Similarity' },
]

export default function App() {
  return (
    <div className="min-h-screen bg-[color:var(--bg)] text-[color:var(--ink)]">
      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <header className="rounded-[36px] border border-[color:var(--border)] bg-white/80 p-5 shadow-[0_18px_48px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.26em] text-slate-500">
                MCF Intelligence
              </p>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight text-[color:var(--ink)]">
                Hiring-market intelligence and explainable NLP retrieval
              </h1>
            </div>

            <nav className="flex flex-wrap gap-2">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  className={({ isActive }) =>
                    `rounded-full px-4 py-2 text-sm font-semibold transition ${
                      isActive
                        ? 'bg-[color:var(--brand)] text-white shadow-lg'
                        : 'bg-[color:var(--surface)] text-slate-700 hover:text-[color:var(--brand)]'
                    }`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
        </header>

        <main className="py-8">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/trends" element={<TrendsPage />} />
            <Route path="/match-lab" element={<MatchLabPage />} />
            <Route path="/search" element={<SearchPage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
