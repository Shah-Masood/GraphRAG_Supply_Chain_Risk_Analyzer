-- ─────────────────────────────────────────────────────────────────────────────
-- Migration 001 — Initial schema
-- Run with: psql $DATABASE_URL -f 001_initial.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- Enum types
CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE risk_category AS ENUM (
    'geopolitical',
    'financial',
    'logistics',
    'environmental',
    'regulatory',
    'supplier_health',
    'cyber'
);
CREATE TYPE event_status AS ENUM ('active', 'monitoring', 'resolved');

-- ─────────────────────────────────────────────────────────────────────────────
-- suppliers
-- Core entity: a company in the supply chain
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE suppliers (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    region      TEXT,                          -- e.g. "Southeast Asia"
    tier        SMALLINT NOT NULL DEFAULT 1,  -- 1 = direct, 2 = sub-supplier
    industry    TEXT,
    metadata    JSONB NOT NULL DEFAULT '{}',  -- flexible extra fields
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_suppliers_country  ON suppliers (country);
CREATE INDEX idx_suppliers_tier     ON suppliers (tier);

-- ─────────────────────────────────────────────────────────────────────────────
-- documents
-- Ingested source documents (PDFs, reports, news articles)
-- Each row = one source; chunks live in ChromaDB referencing doc_id
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_url      TEXT,
    title           TEXT,
    doc_type        TEXT NOT NULL,          -- 'pdf' | 'news' | 'report' | 'web'
    supplier_id     UUID REFERENCES suppliers (id) ON DELETE SET NULL,
    country         TEXT,
    raw_text_hash   TEXT,                   -- SHA-256 of raw content, for dedup
    chunk_count     INT NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_supplier    ON documents (supplier_id);
CREATE INDEX idx_documents_doc_type    ON documents (doc_type);
CREATE INDEX idx_documents_ingested_at ON documents (ingested_at DESC);
CREATE UNIQUE INDEX idx_documents_hash ON documents (raw_text_hash)
    WHERE raw_text_hash IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────────
-- risk_events
-- A discrete risk event identified by the agent or ingestion pipeline
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE risk_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    description     TEXT,
    category        risk_category NOT NULL,
    level           risk_level NOT NULL,
    status          event_status NOT NULL DEFAULT 'active',
    supplier_id     UUID REFERENCES suppliers (id) ON DELETE SET NULL,
    document_id     UUID REFERENCES documents (id) ON DELETE SET NULL,
    country         TEXT,
    region          TEXT,
    source_url      TEXT,
    occurred_at     TIMESTAMPTZ,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_risk_events_supplier   ON risk_events (supplier_id);
CREATE INDEX idx_risk_events_category   ON risk_events (category);
CREATE INDEX idx_risk_events_level      ON risk_events (level);
CREATE INDEX idx_risk_events_status     ON risk_events (status);
CREATE INDEX idx_risk_events_detected   ON risk_events (detected_at DESC);
-- GIN index for searching inside metadata
CREATE INDEX idx_risk_events_metadata   ON risk_events USING GIN (metadata);

-- ─────────────────────────────────────────────────────────────────────────────
-- risk_scores
-- Computed aggregate risk scores per supplier, updated by the agent
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE risk_scores (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id     UUID NOT NULL REFERENCES suppliers (id) ON DELETE CASCADE,
    overall_score   NUMERIC(5, 2) NOT NULL CHECK (overall_score BETWEEN 0 AND 100),
    -- Per-category breakdown (nullable if not assessed)
    geopolitical    NUMERIC(5, 2),
    financial       NUMERIC(5, 2),
    logistics       NUMERIC(5, 2),
    environmental   NUMERIC(5, 2),
    regulatory      NUMERIC(5, 2),
    supplier_health NUMERIC(5, 2),
    cyber           NUMERIC(5, 2),
    reasoning       TEXT,                   -- agent's explanation
    model_version   TEXT,                   -- which LLM produced this
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_risk_scores_supplier ON risk_scores (supplier_id);
CREATE INDEX idx_risk_scores_scored   ON risk_scores (scored_at DESC);

-- Most recent score per supplier (useful view)
CREATE VIEW latest_risk_scores AS
SELECT DISTINCT ON (supplier_id)
    rs.*,
    s.name  AS supplier_name,
    s.country,
    s.tier
FROM risk_scores rs
JOIN suppliers s ON s.id = rs.supplier_id
ORDER BY supplier_id, scored_at DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- agent_sessions
-- Maps LangGraph thread_ids to human context (who asked, about what)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE agent_sessions (
    thread_id       TEXT PRIMARY KEY,       -- LangGraph checkpoint thread_id
    user_id         TEXT,
    supplier_id     UUID REFERENCES suppliers (id) ON DELETE SET NULL,
    topic           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_sessions_user     ON agent_sessions (user_id);
CREATE INDEX idx_agent_sessions_supplier ON agent_sessions (supplier_id);

-- Auto-update updated_at on suppliers
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER suppliers_updated_at
    BEFORE UPDATE ON suppliers
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
