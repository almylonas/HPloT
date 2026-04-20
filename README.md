# HPlot — Invariant Mass Explorer

A web app for visualising LHC invariant mass datasets created in Hypatia, backed by Neon PostgreSQL.

---

## 1. Set up Neon database

1. Create a free project at [neon.tech](https://neon.tech)
2. Open the **SQL Editor** in your Neon dashboard
3. Paste and run the contents of `sql/schema.sql`
4. Copy your **Connection string** (it looks like `postgres://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`)

---

## 2. Run locally

```bash
npm install
cp .env.local.example .env.local
# Paste your Neon connection string into .env.local
npm run dev
```

Open http://localhost:3000

---

## Data format

Upload `.csv` or `.txt` files with three columns (no header):

```
invariant_mass, particle_type, combination
91.2, 1, e
88.5, 2, m
124.0, 3, g
1500.0, 4, 4mm
```

| Column | Values |
|--------|--------|
| `invariant_mass` | float, GeV |
| `particle_type` | 1=electron, 2=muon, 3=photon, 4=four-lepton |
| `combination` | `e`, `m`, `g`, `4ee`, `4mm`, `4me` |

---

## Features

- Upload multiple datasets — all stored in Neon
- **Cumulative mode**: analyse all uploaded datasets together
- **Selected mode**: pick specific datasets to overlay
- Dilepton, four-lepton, and diphoton histograms
- Adjustable bins, log X-axis
- Energy-range statistics table per particle type
