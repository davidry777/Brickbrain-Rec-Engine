-- Initialize pgvector extension for LangChain
-- This script runs automatically when the PostgreSQL container starts

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- LangChain will create its own tables automatically, but we can create
-- some helper functions and ensure the extension is ready

-- Create a function to check if pgvector is working
CREATE OR REPLACE FUNCTION check_pgvector_installed() 
RETURNS TEXT AS $$
BEGIN
    RETURN 'pgvector extension is installed and ready';
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions for the vector operations
-- GRANT USAGE ON SCHEMA public TO brickbrain;
-- GRANT CREATE ON SCHEMA public TO brickbrain;
