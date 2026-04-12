-- HPlot Database Schema for Neon PostgreSQL
-- Run this in your Neon SQL editor to set up the database

CREATE TABLE IF NOT EXISTS datasets (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    row_count   INTEGER NOT NULL DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id             BIGSERIAL PRIMARY KEY,
    dataset_id     INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    invariant_mass DOUBLE PRECISION NOT NULL,
    particle_type  INTEGER NOT NULL,       -- 1=electron, 2=muon, 3=photon, 4=four-lepton
    combination    TEXT NOT NULL DEFAULT '' -- 'e','m','g','4ee','4mm','4me', etc.
);

-- Index for fast filtering
CREATE INDEX IF NOT EXISTS idx_events_dataset    ON events(dataset_id);
CREATE INDEX IF NOT EXISTS idx_events_combination ON events(combination);
CREATE INDEX IF NOT EXISTS idx_events_particle    ON events(particle_type);
CREATE INDEX IF NOT EXISTS idx_events_mass        ON events(invariant_mass);
