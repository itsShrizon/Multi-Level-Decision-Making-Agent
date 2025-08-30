-- Initialize database for development
-- This file is used by Docker to set up the initial database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create enum types for consistency
CREATE TYPE sentiment_type AS ENUM ('Positive', 'Neutral', 'Negative');
CREATE TYPE risk_level AS ENUM ('Low', 'Medium', 'High');
CREATE TYPE message_action AS ENUM ('FLAG', 'IGNORE', 'RESPOND');

-- Future tables will be created via Alembic migrations
-- This is just for basic setup
