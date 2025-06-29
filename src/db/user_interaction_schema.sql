-- Users table for storing user profiles
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- User preferences
    preferred_themes INTEGER[], -- Array of theme IDs
    preferred_min_pieces INTEGER DEFAULT 0,
    preferred_max_pieces INTEGER DEFAULT 10000,
    preferred_min_year INTEGER DEFAULT 1950,
    preferred_max_year INTEGER DEFAULT 2030,
    complexity_preference VARCHAR(20) DEFAULT 'moderate', -- 'simple', 'moderate', 'complex'
    budget_range_min DECIMAL(10,2) DEFAULT 0.00,
    budget_range_max DECIMAL(10,2) DEFAULT 1000.00,
    
    -- Profile stats
    total_ratings INTEGER DEFAULT 0,
    avg_rating DECIMAL(3,2) DEFAULT 0.00,
    total_sets_owned INTEGER DEFAULT 0
);

-- User interactions table (ratings, views, purchases, wishlists)
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_num VARCHAR(20) NOT NULL REFERENCES sets(set_num) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL, -- 'rating', 'view', 'purchase', 'wishlist', 'click'
    rating INTEGER CHECK (rating BETWEEN 1 AND 5), -- Only for rating interactions
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional context
    session_id VARCHAR(255),
    source VARCHAR(50), -- 'recommendation', 'search', 'browse', 'external'
    
    UNIQUE(user_id, set_num, interaction_type) -- Prevent duplicate interactions of same type
);

-- User collections (sets they own)
CREATE TABLE IF NOT EXISTS user_collections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_num VARCHAR(20) NOT NULL REFERENCES sets(set_num) ON DELETE CASCADE,
    acquisition_date DATE DEFAULT CURRENT_DATE,
    purchase_price DECIMAL(10,2),
    condition VARCHAR(20) DEFAULT 'new', -- 'new', 'used', 'incomplete'
    is_built BOOLEAN DEFAULT FALSE,
    is_displayed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    
    UNIQUE(user_id, set_num)
);

-- User wishlists
CREATE TABLE IF NOT EXISTS user_wishlists (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_num VARCHAR(20) NOT NULL REFERENCES sets(set_num) ON DELETE CASCADE,
    priority INTEGER DEFAULT 3, -- 1 (highest) to 5 (lowest)
    max_price DECIMAL(10,2), -- Maximum price willing to pay
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    UNIQUE(user_id, set_num)
);

-- Recommendation cache table
CREATE TABLE IF NOT EXISTS user_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_num VARCHAR(20) NOT NULL REFERENCES sets(set_num) ON DELETE CASCADE,
    recommendation_type VARCHAR(20) NOT NULL, -- 'content', 'collaborative', 'hybrid'
    score DECIMAL(5,4) NOT NULL,
    reasons JSONB, -- Store reasons as JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
    is_shown BOOLEAN DEFAULT FALSE, -- Track if recommendation was actually shown
    clicked BOOLEAN DEFAULT FALSE, -- Track if user clicked on it
    
    UNIQUE(user_id, set_num, recommendation_type)
);

-- User reviews table (detailed feedback beyond ratings)
CREATE TABLE IF NOT EXISTS user_reviews (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    set_num VARCHAR(20) NOT NULL REFERENCES sets(set_num) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title VARCHAR(255),
    review_text TEXT,
    pros TEXT[],
    cons TEXT[],
    difficulty_rating INTEGER CHECK (difficulty_rating BETWEEN 1 AND 5),
    value_rating INTEGER CHECK (value_rating BETWEEN 1 AND 5),
    design_rating INTEGER CHECK (design_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_verified_purchase BOOLEAN DEFAULT FALSE,
    helpful_votes INTEGER DEFAULT 0,
    
    UNIQUE(user_id, set_num)
);

-- Set analytics table (computed metrics)
CREATE TABLE IF NOT EXISTS set_analytics (
    set_num VARCHAR(20) PRIMARY KEY REFERENCES sets(set_num) ON DELETE CASCADE,
    
    -- Rating analytics
    avg_rating DECIMAL(3,2) DEFAULT 0.00,
    total_ratings INTEGER DEFAULT 0,
    rating_distribution JSONB, -- {1: count, 2: count, ...}
    
    -- Interaction analytics
    total_views INTEGER DEFAULT 0,
    total_clicks INTEGER DEFAULT 0,
    total_wishlists INTEGER DEFAULT 0,
    conversion_rate DECIMAL(5,4) DEFAULT 0.0000, -- clicks to purchases
    
    -- Recommendation analytics
    recommendation_frequency INTEGER DEFAULT 0,
    recommendation_click_rate DECIMAL(5,4) DEFAULT 0.0000,
    
    -- Computed features for ML
    popularity_score DECIMAL(5,4) DEFAULT 0.0000,
    trending_score DECIMAL(5,4) DEFAULT 0.0000,
    complexity_score DECIMAL(5,4) DEFAULT 0.0000,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User similarity table (for collaborative filtering optimization)
CREATE TABLE IF NOT EXISTS user_similarities (
    user_id_1 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_id_2 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5,4) NOT NULL,
    similarity_type VARCHAR(20) NOT NULL, -- 'cosine', 'pearson', 'jaccard'
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (user_id_1, user_id_2, similarity_type),
    CHECK (user_id_1 < user_id_2) -- Ensure no duplicate pairs
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_set_num ON user_interactions(set_num);
CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_user_interactions_rating ON user_interactions(rating) WHERE rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_user_interactions_created_at ON user_interactions(created_at);

CREATE INDEX IF NOT EXISTS idx_user_collections_user_id ON user_collections(user_id);
CREATE INDEX IF NOT EXISTS idx_user_wishlists_user_id ON user_wishlists(user_id);
CREATE INDEX IF NOT EXISTS idx_user_reviews_user_id ON user_reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_user_reviews_set_num ON user_reviews(set_num);
CREATE INDEX IF NOT EXISTS idx_user_reviews_rating ON user_reviews(rating);

CREATE INDEX IF NOT EXISTS idx_user_recommendations_user_id ON user_recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_recommendations_expires_at ON user_recommendations(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_recommendations_score ON user_recommendations(score);

CREATE INDEX IF NOT EXISTS idx_set_analytics_avg_rating ON set_analytics(avg_rating);
CREATE INDEX IF NOT EXISTS idx_set_analytics_popularity ON set_analytics(popularity_score);
CREATE INDEX IF NOT EXISTS idx_set_analytics_trending ON set_analytics(trending_score);

-- Create function to update user stats automatically
CREATE OR REPLACE FUNCTION update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update user rating statistics
    UPDATE users 
    SET 
        total_ratings = (
            SELECT COUNT(*) 
            FROM user_interactions 
            WHERE user_id = NEW.user_id AND interaction_type = 'rating'
        ),
        avg_rating = (
            SELECT AVG(rating::DECIMAL) 
            FROM user_interactions 
            WHERE user_id = NEW.user_id AND interaction_type = 'rating' AND rating IS NOT NULL
        )
    WHERE id = NEW.user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update user stats
DROP TRIGGER IF EXISTS trigger_update_user_stats ON user_interactions;
CREATE TRIGGER trigger_update_user_stats
    AFTER INSERT OR UPDATE ON user_interactions
    FOR EACH ROW
    EXECUTE FUNCTION update_user_stats();

-- Create function to update set analytics
CREATE OR REPLACE FUNCTION update_set_analytics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update set rating statistics
    INSERT INTO set_analytics (set_num, avg_rating, total_ratings, total_views, updated_at)
    VALUES (NEW.set_num, 0, 0, 0, CURRENT_TIMESTAMP)
    ON CONFLICT (set_num) DO UPDATE SET
        avg_rating = (
            SELECT AVG(rating::DECIMAL) 
            FROM user_interactions 
            WHERE set_num = NEW.set_num AND interaction_type = 'rating' AND rating IS NOT NULL
        ),
        total_ratings = (
            SELECT COUNT(*) 
            FROM user_interactions 
            WHERE set_num = NEW.set_num AND interaction_type = 'rating'
        ),
        total_views = (
            SELECT COUNT(*) 
            FROM user_interactions 
            WHERE set_num = NEW.set_num AND interaction_type = 'view'
        ),
        total_clicks = (
            SELECT COUNT(*) 
            FROM user_interactions 
            WHERE set_num = NEW.set_num AND interaction_type = 'click'
        ),
        total_wishlists = (
            SELECT COUNT(*) 
            FROM user_wishlists 
            WHERE set_num = NEW.set_num
        ),
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update set analytics
DROP TRIGGER IF EXISTS trigger_update_set_analytics ON user_interactions;
CREATE TRIGGER trigger_update_set_analytics
    AFTER INSERT OR UPDATE ON user_interactions
    FOR EACH ROW
    EXECUTE FUNCTION update_set_analytics();

-- Insert some sample users for testing
INSERT INTO users (username, email, password_hash, preferred_themes, complexity_preference) VALUES
('brickmaster', 'brickmaster@example.com', 'hashed_password_1', ARRAY[1, 2, 3], 'complex'),
('casual_builder', 'casual@example.com', 'hashed_password_2', ARRAY[4, 5], 'simple'),
('space_fan', 'spacefan@example.com', 'hashed_password_3', ARRAY[143, 273], 'moderate'),
('city_lover', 'citylover@example.com', 'hashed_password_4', ARRAY[1, 52], 'moderate'),
('technic_pro', 'technicpro@example.com', 'hashed_password_5', ARRAY[1, 4], 'complex')
ON CONFLICT (username) DO NOTHING;

-- Create a view for easy recommendation analysis
CREATE OR REPLACE VIEW recommendation_performance AS
SELECT 
    ur.recommendation_type,
    COUNT(*) as total_recommendations,
    COUNT(*) FILTER (WHERE ur.is_shown = true) as shown_recommendations,
    COUNT(*) FILTER (WHERE ur.clicked = true) as clicked_recommendations,
    ROUND(
        COUNT(*) FILTER (WHERE ur.clicked = true)::DECIMAL / 
        NULLIF(COUNT(*) FILTER (WHERE ur.is_shown = true), 0) * 100, 2
    ) as click_through_rate,
    AVG(ur.score) as avg_score
FROM user_recommendations ur
WHERE ur.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY ur.recommendation_type;

-- Create materialized view for popular sets (refresh periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS popular_sets AS
SELECT 
    s.set_num,
    s.name,
    s.year,
    s.theme_id,
    t.name as theme_name,
    sa.avg_rating,
    sa.total_ratings,
    sa.total_views,
    sa.popularity_score,
    ROW_NUMBER() OVER (ORDER BY sa.popularity_score DESC, sa.avg_rating DESC) as popularity_rank
FROM sets s
JOIN themes t ON s.theme_id = t.id
LEFT JOIN set_analytics sa ON s.set_num = sa.set_num
WHERE sa.total_ratings >= 3  -- Minimum number of ratings
ORDER BY sa.popularity_score DESC, sa.avg_rating DESC;

-- Create indexes on the materialized view
CREATE INDEX IF NOT EXISTS idx_popular_sets_rank ON popular_sets(popularity_rank);
CREATE INDEX IF NOT EXISTS idx_popular_sets_theme ON popular_sets(theme_id);
CREATE INDEX IF NOT EXISTS idx_popular_sets_year ON popular_sets(year);

-- Comment on tables
COMMENT ON TABLE users IS 'User profiles and preferences for the recommendation system';
COMMENT ON TABLE user_interactions IS 'All user interactions with sets (ratings, views, clicks, etc.)';
COMMENT ON TABLE user_collections IS 'Sets that users own in their collection';
COMMENT ON TABLE user_wishlists IS 'Sets that users want to acquire';
COMMENT ON TABLE user_recommendations IS 'Generated recommendations cache with performance tracking';
COMMENT ON TABLE user_reviews IS 'Detailed user reviews and ratings for sets';
COMMENT ON TABLE set_analytics IS 'Computed analytics and metrics for each set';
COMMENT ON TABLE user_similarities IS 'Precomputed user similarity scores for collaborative filtering';