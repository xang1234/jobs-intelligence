# MCF Intelligence Frontend

The frontend is a React + Vite application for the hiring-market intelligence experience. It is organized around four recruiter-oriented workflows:

- `Overview`: headline metrics, rising skills, rising companies, and salary movement
- `Trends Explorer`: monthly comparisons for skills, roles, and companies
- `Match Lab`: pasted profile-to-job matching with explicit fit breakdowns
- `Search & Similarity`: hybrid search, query-expansion inspector, related-skills graph, and similar jobs

## Local Development

```bash
cd src/frontend
npm install
npm run dev
```

The backend API is expected at `/api/*` on the same origin. During local development, run the FastAPI server separately from the repo root:

```bash
poetry run python -m src.cli api-serve --reload
```

## Build

```bash
cd src/frontend
npm run build
```

## Product Notes

- The UI intentionally exposes explanation payloads from the backend instead of hiding the ranking logic.
- Trend charts are lightweight SVG components so the app does not need a dedicated charting dependency.
- The visual system is tuned for an intelligence-product feel rather than a default search/admin layout.
