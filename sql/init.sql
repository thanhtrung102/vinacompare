-- VinaCompare Database Initialization Script

-- Query logs
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    user_id VARCHAR(50),
    query_text TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    session_id UUID,
    ip_address VARCHAR(45)
);

-- Model responses
CREATE TABLE IF NOT EXISTS model_responses (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,
    answer TEXT NOT NULL,
    confidence_score FLOAT,
    hallucination_score FLOAT,
    latency_ms INTEGER,
    tokens_used INTEGER,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Retrieval results
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    hit_rate_1 FLOAT,
    hit_rate_5 FLOAT,
    hit_rate_10 FLOAT,
    mrr FLOAT,
    ndcg_5 FLOAT,
    ndcg_10 FLOAT,
    search_time_ms INTEGER,
    num_results INTEGER,
    retrieval_mode VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User feedback
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    model_name VARCHAR(50),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    thumbs_direction VARCHAR(10) CHECK (thumbs_direction IN ('up', 'down')),
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- LLM judge evaluations
CREATE TABLE IF NOT EXISTS llm_judge_scores (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    model_name VARCHAR(50),
    accuracy_score FLOAT CHECK (accuracy_score BETWEEN 0 AND 5),
    completeness_score FLOAT CHECK (completeness_score BETWEEN 0 AND 5),
    language_quality_score FLOAT CHECK (language_quality_score BETWEEN 0 AND 5),
    relevance_score FLOAT CHECK (relevance_score BETWEEN 0 AND 5),
    usefulness_score FLOAT CHECK (usefulness_score BETWEEN 0 AND 5),
    total_score FLOAT CHECK (total_score BETWEEN 0 AND 25),
    explanation TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Document metadata
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(100) UNIQUE NOT NULL,
    title TEXT,
    content TEXT,
    source_url TEXT,
    source_type VARCHAR(50),
    language VARCHAR(10) DEFAULT 'vi',
    category VARCHAR(50),
    tags TEXT[],
    published_at TIMESTAMP,
    ingested_at TIMESTAMP DEFAULT NOW(),
    chunk_count INTEGER,
    token_count INTEGER
);

-- Ground truth dataset
CREATE TABLE IF NOT EXISTS ground_truth (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    document_id VARCHAR(100) REFERENCES documents(document_id),
    difficulty VARCHAR(20) CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    category VARCHAR(50),
    manually_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX idx_model_responses_query_id ON model_responses(query_id);
CREATE INDEX idx_model_responses_model_name ON model_responses(model_name);
CREATE INDEX idx_model_responses_created_at ON model_responses(created_at);
CREATE INDEX idx_feedback_query_id ON feedback(query_id);
CREATE INDEX idx_feedback_model_name ON feedback(model_name);
CREATE INDEX idx_retrieval_logs_query_id ON retrieval_logs(query_id);
CREATE INDEX idx_documents_document_id ON documents(document_id);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_ground_truth_category ON ground_truth(category);
CREATE INDEX idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Materialized view for model performance
CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_summary AS
SELECT
    mr.model_name,
    COUNT(DISTINCT mr.query_id) as total_queries,
    AVG(mr.confidence_score) as avg_confidence,
    AVG(mr.hallucination_score) as avg_hallucination,
    AVG(mr.latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY mr.latency_ms) as p95_latency_ms,
    SUM(mr.cost_usd) as total_cost_usd,
    AVG(ljs.total_score) as avg_judge_score,
    COUNT(f.id) FILTER (WHERE f.thumbs_direction = 'up')::FLOAT /
        NULLIF(COUNT(f.id), 0) as satisfaction_rate,
    MAX(mr.created_at) as last_query_at
FROM model_responses mr
LEFT JOIN llm_judge_scores ljs
    ON mr.query_id = ljs.query_id AND mr.model_name = ljs.model_name
LEFT JOIN feedback f
    ON mr.query_id = f.query_id AND mr.model_name = f.model_name
WHERE mr.created_at > NOW() - INTERVAL '7 days'
GROUP BY mr.model_name;

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_model_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW model_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vinarag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vinarag;

-- Insert initial system status
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, tags)
VALUES ('system_initialized', 1, 'boolean', '{"version": "1.0.0"}');

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'VinaCompare database initialized successfully!';
END $$;
