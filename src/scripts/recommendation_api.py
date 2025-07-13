from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime
import os
from contextlib import contextmanager, asynccontextmanager
import asyncio
import time
from functools import wraps

# Import our recommendation system
from recommendation_system import HybridRecommender, RecommendationResult, UserProfile
from lego_nlp_recommeder import NLPRecommender, NLQueryResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_PARAMS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "brickbrain"),
    "user": os.getenv("DB_USER", "brickbrain"),
    "password": os.getenv("DB_PASSWORD", "brickbrain_password"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# Global recommendation engine instance
recommendation_engine: Optional[HybridRecommender] = None
nl_recommender: Optional[NLPRecommender] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global recommendation_engine, nl_recommender
    
    # Check if we should skip heavy initialization for faster startup
    skip_heavy_init = os.getenv("SKIP_HEAVY_INITIALIZATION", "false").lower() == "true"
    
    # Startup
    try:
        # Create a new connection that will be used by the recommendation engine
        conn = psycopg2.connect(**DB_PARAMS)
        recommendation_engine = HybridRecommender(conn)
        
        if not skip_heavy_init:
            # Initialize the content-based recommender features
            logger.info("Preparing content-based features...")
            recommendation_engine.content_recommender.prepare_features()
            
            # Initialize collaborative filtering (prepare user-item matrix)
            logger.info("Preparing collaborative filtering...")
            recommendation_engine.collaborative_recommender.prepare_user_item_matrix()
            
            logger.info("Recommendation engine initialized successfully")

            # Initialize NLP recommender
            logger.info("Initializing natural language processor...")
            nl_recommender = NLPRecommender(conn, use_openai=False)

            # Load or create embeddings
            embeddings_path = "./embeddings/faiss_index"
            if os.path.exists(embeddings_path):
                logger.info("Loading existing embeddings...")
                if not nl_recommender.load_embeddings(embeddings_path):
                    logger.warning("Failed to load existing embeddings, creating new ones...")
                    # Start with a smaller dataset for faster initialization
                    logger.info("Processing top 1000 sets by popularity for initial setup...")
                    nl_recommender.prep_vectorDB(limit_sets=1000)
                    os.makedirs("./embeddings", exist_ok=True)
                    nl_recommender.save_embeddings(embeddings_path)
                    logger.info("Initial embeddings created successfully")
            else:
                logger.info("Creating new embeddings (this may take a few minutes)...")
                # Start with a smaller dataset for faster initialization
                logger.info("Processing top 1000 sets by popularity for initial setup...")
                nl_recommender.prep_vectorDB(limit_sets=1000)
                os.makedirs("./embeddings", exist_ok=True)
                nl_recommender.save_embeddings(embeddings_path)
                logger.info("Initial embeddings created successfully")
        else:
            logger.info("Skipping heavy initialization for faster startup")
            logger.info("Basic recommendation engine initialized successfully")
            
            # Initialize NLP recommender with minimal setup
            logger.info("Initializing NLP recommender with minimal setup...")
            nl_recommender = NLPRecommender(conn, use_openai=False)
            logger.info("NLP recommender initialized (without embeddings)")
            
        logger.info("API server startup complete")

        logger.info("Natural language processor initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        # Don't raise the exception to allow the API to start without recommendations
        logger.warning("API will start without recommendation capabilities")
    
    yield
    
    # Shutdown
    logger.info("API shutting down...")
    if recommendation_engine and recommendation_engine.dbcon:
        try:
            recommendation_engine.dbcon.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

# FastAPI app configuration
app = FastAPI(
    title="LEGO Recommendation Engine API",
    description="A production-ready API for LEGO set recommendations using ML algorithms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection management
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_db():
    """Dependency for getting database connection"""
    with get_db_connection() as conn:
        yield conn

# Pydantic models for request/response
class UserInteractionRequest(BaseModel):
    user_id: int
    set_num: str
    interaction_type: str = Field(..., pattern="^(rating|view|purchase|wishlist|click)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    source: Optional[str] = "api"
    session_id: Optional[str] = None

class UserPreferencesRequest(BaseModel):
    user_id: int
    preferred_themes: Optional[List[int]] = []
    preferred_min_pieces: Optional[int] = 0
    preferred_max_pieces: Optional[int] = 10000
    preferred_min_year: Optional[int] = 1950
    preferred_max_year: Optional[int] = 2030
    complexity_preference: Optional[str] = Field("moderate", pattern="^(simple|moderate|complex)$")
    budget_range_min: Optional[float] = 0.0
    budget_range_max: Optional[float] = 1000.0

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    set_num: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)
    recommendation_type: str = Field("hybrid", pattern="^(content|collaborative|hybrid)$")
    include_reasons: bool = True

class SetSearchRequest(BaseModel):
    query: Optional[str] = None
    theme_ids: Optional[List[int]] = []
    min_pieces: Optional[int] = None
    max_pieces: Optional[int] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_rating: Optional[float] = None
    sort_by: str = Field("name", pattern="^(name|year|num_parts|rating)$")
    sort_order: str = Field("asc", pattern="^(asc|desc)$")
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(..., min_length=6)

class RecommendationResponse(BaseModel):
    set_num: str
    name: str
    score: float
    reasons: List[str]
    theme_name: str
    year: int
    num_parts: int
    img_url: Optional[str]

class UserProfileResponse(BaseModel):
    user_id: int
    username: str
    total_ratings: int
    avg_rating: float
    total_sets_owned: int
    preferred_themes: List[int]
    complexity_preference: str

class SetDetailsResponse(BaseModel):
    set_num: str
    name: str
    year: int
    theme_name: str
    num_parts: int
    img_url: Optional[str]
    avg_rating: Optional[float]
    total_ratings: int
    is_in_user_collection: bool = False
    is_in_user_wishlist: bool = False

class APIMetrics(BaseModel):
    total_requests: int
    avg_response_time: float
    active_users: int
    total_recommendations_served: int
    cache_hit_rate: float

class NaturalLanguageQuery(BaseModel):
    query: str = Field(..., description="Natural language query describing desired LEGO sets")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    include_explanation: bool = Field(True, description="Include natural language explanation")

class NLSearchResult(BaseModel):
    set_num: str
    name: str
    theme: str
    year: int
    num_parts: int
    relevance_score: float
    match_reasons: List[str]
    description: str

class NLSearchResponse(BaseModel):
    query: str
    intent: str
    extracted_filters: Dict[str, Any]
    results: List[NLSearchResult]
    explanation: Optional[str]
    query_understanding: Dict[str, Any]

class SimilarSetQuery(BaseModel):
    set_num: str = Field(..., description="Set number to find similar sets for")
    description: Optional[str] = Field(None, description="Additional context or requirements")
    top_k: int = Field(10, ge=1, le=50)

class ConversationalQuery(BaseModel):
    query: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    user_id: Optional[int] = None
    context: Dict[str, Any] = Field(default_factory=dict)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.recommendation_count = 0
        self.cache_hits = 0
        self.cache_requests = 0

    def record_request(self, response_time: float):
        self.request_count += 1
        self.total_response_time += response_time

    def record_recommendation(self):
        self.recommendation_count += 1

    def record_cache_hit(self, hit: bool):
        self.cache_requests += 1
        if hit:
            self.cache_hits += 1

    def get_metrics(self) -> APIMetrics:
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_requests, 1)
        
        return APIMetrics(
            total_requests=self.request_count,
            avg_response_time=avg_response_time,
            active_users=0,  # Would need session tracking
            total_recommendations_served=self.recommendation_count,
            cache_hit_rate=cache_hit_rate
        )

performance_monitor = PerformanceMonitor()

# Timing decorator
def time_request(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            response_time = time.time() - start_time
            performance_monitor.record_request(response_time)
    return wrapper

# Core recommendation endpoints
@app.post("/recommendations", response_model=List[RecommendationResponse])
@time_request
async def get_recommendations(
    request: RecommendationRequest,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get personalized recommendations for a user or similar sets for a given set
    """
    try:
        if not recommendation_engine:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")

        # Check cache first
        cached_recs = _get_cached_recommendations(db, request)
        if cached_recs:
            performance_monitor.record_cache_hit(True)
            performance_monitor.record_recommendation()
            return cached_recs

        performance_monitor.record_cache_hit(False)

        # Generate new recommendations
        if request.recommendation_type == "content" and request.set_num:
            recommendations = recommendation_engine.content_recommender.get_similar_sets(
                request.set_num, request.top_k
            )
        elif request.recommendation_type == "collaborative" and request.user_id:
            recommendations = recommendation_engine.collaborative_recommender.get_recommendations(
                str(request.user_id), request.top_k
            )
        elif request.recommendation_type == "hybrid":
            recommendations = recommendation_engine.get_recommendations(
                user_id=request.user_id,
                liked_set=request.set_num,
                top_k=request.top_k
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid recommendation type or missing required parameters"
            )

        # Convert to response format
        response_recs = [
            RecommendationResponse(
                set_num=rec.set_num,
                name=rec.name,
                score=rec.score,
                reasons=rec.reasons if request.include_reasons else [],
                theme_name=rec.theme_name,
                year=rec.year,
                num_parts=rec.num_parts,
                img_url=rec.img_url
            )
            for rec in recommendations
        ]

        # Cache the recommendations
        _cache_recommendations(db, request, response_recs)
        
        performance_monitor.record_recommendation()
        return response_recs

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/interactions")
@time_request
async def track_user_interaction(
    user_id: int,
    interaction: UserInteractionRequest,
    background_tasks: BackgroundTasks,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Track user interactions (ratings, views, purchases, etc.)
    """
    try:
        # Validate user exists
        if not _user_exists(db, user_id):
            raise HTTPException(status_code=404, detail="User not found")

        # Validate set exists
        if not _set_exists(db, interaction.set_num):
            raise HTTPException(status_code=404, detail="Set not found")

        # Insert interaction
        cursor = db.cursor()
        query = """
        INSERT INTO user_interactions (user_id, set_num, interaction_type, rating, source, session_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id, set_num, interaction_type) 
        DO UPDATE SET 
            rating = EXCLUDED.rating,
            created_at = CURRENT_TIMESTAMP,
            source = EXCLUDED.source,
            session_id = EXCLUDED.session_id
        """
        
        cursor.execute(query, (
            user_id,
            interaction.set_num,
            interaction.interaction_type,
            interaction.rating,
            interaction.source,
            interaction.session_id
        ))
        db.commit()

        # Update user stats if it's a rating
        if interaction.interaction_type == "rating" and interaction.rating:
            background_tasks.add_task(_update_user_rating_stats, user_id, db.dsn)

        return {"message": "Interaction tracked successfully"}

    except Exception as e:
        logger.error(f"Error tracking interaction: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users", response_model=dict)
@time_request
async def create_user(
    user_data: UserRegistrationRequest,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Create a new user account
    """
    try:
        import hashlib
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        cursor = db.cursor()
        query = """
        INSERT INTO users (username, email, password_hash)
        VALUES (%s, %s, %s)
        RETURNING id
        """
        
        cursor.execute(query, (user_data.username, user_data.email, password_hash))
        user_id = cursor.fetchone()[0]
        db.commit()
        
        return {"user_id": user_id, "message": "User created successfully"}

    except psycopg2.IntegrityError as e:
        db.rollback()
        if "username" in str(e):
            raise HTTPException(status_code=400, detail="Username already exists")
        elif "email" in str(e):
            raise HTTPException(status_code=400, detail="Email already exists")
        else:
            raise HTTPException(status_code=400, detail="User creation failed")
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/profile", response_model=UserProfileResponse)
@time_request
async def get_user_profile(
    user_id: int,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get user profile information
    """
    try:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        query = """
        SELECT u.id as user_id, u.username, u.total_ratings, u.avg_rating,
               u.total_sets_owned, u.preferred_themes, u.complexity_preference
        FROM users u
        WHERE u.id = %s
        """
        
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserProfileResponse(**user)

    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/{user_id}/preferences")
@time_request
async def update_user_preferences(
    user_id: int,
    preferences: UserPreferencesRequest,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Update user preferences
    """
    try:
        if not _user_exists(db, user_id):
            raise HTTPException(status_code=404, detail="User not found")

        cursor = db.cursor()
        query = """
        UPDATE users SET
            preferred_themes = %s,
            preferred_min_pieces = %s,
            preferred_max_pieces = %s,
            preferred_min_year = %s,
            preferred_max_year = %s,
            complexity_preference = %s,
            budget_range_min = %s,
            budget_range_max = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = %s
        """
        
        cursor.execute(query, (
            preferences.preferred_themes,
            preferences.preferred_min_pieces,
            preferences.preferred_max_pieces,
            preferences.preferred_min_year,
            preferences.preferred_max_year,
            preferences.complexity_preference,
            preferences.budget_range_min,
            preferences.budget_range_max,
            user_id
        ))
        db.commit()
        
        return {"message": "User preferences updated successfully"}

    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/sets", response_model=List[SetDetailsResponse])
@time_request
async def search_sets(
    search_request: SetSearchRequest,
    user_id: Optional[int] = Query(None),
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Search for LEGO sets with filtering and sorting
    """
    try:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        # Build dynamic query
        where_conditions = []
        params = []
        
        # Always filter out accessory sets (num_parts > 0 ensures we get actual building sets)
        where_conditions.append("s.num_parts > 0")
        
        # Exclude common accessory/gear themes that aren't building sets
        accessory_themes = [501, 503, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 777]
        where_conditions.append("s.theme_id NOT IN %s")
        params.append(tuple(accessory_themes))
        
        if search_request.query:
            where_conditions.append("(s.name ILIKE %s OR t.name ILIKE %s)")
            params.extend([f"%{search_request.query}%", f"%{search_request.query}%"])
        
        if search_request.theme_ids:
            where_conditions.append("s.theme_id = ANY(%s)")
            params.append(search_request.theme_ids)
        
        if search_request.min_pieces is not None:
            where_conditions.append("s.num_parts >= %s")
            params.append(search_request.min_pieces)
        
        if search_request.max_pieces is not None:
            where_conditions.append("s.num_parts <= %s")
            params.append(search_request.max_pieces)
        
        if search_request.min_year is not None:
            where_conditions.append("s.year >= %s")
            params.append(search_request.min_year)
        
        if search_request.max_year is not None:
            where_conditions.append("s.year <= %s")
            params.append(search_request.max_year)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Order clause
        order_direction = "DESC" if search_request.sort_order == "desc" else "ASC"
        order_clause = f"ORDER BY s.{search_request.sort_by} {order_direction}"
        
        # Add user_id parameters at the beginning for the EXISTS queries
        user_params = [user_id or 0, user_id or 0]  # Use 0 if user_id is None
        all_params = user_params + params
        
        query = f"""
        SELECT s.set_num, s.name, s.year, t.name as theme_name, s.num_parts, s.img_url,
               COALESCE(AVG(ui.rating), 0) as avg_rating,
               COUNT(ui.rating) as total_ratings,
               EXISTS(
                   SELECT 1 FROM user_collections uc 
                   WHERE uc.user_id = %s AND uc.set_num = s.set_num
               ) as is_in_user_collection,
               EXISTS(
                   SELECT 1 FROM user_wishlists uw 
                   WHERE uw.user_id = %s AND uw.set_num = s.set_num
               ) as is_in_user_wishlist
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN user_interactions ui ON s.set_num = ui.set_num AND ui.interaction_type = 'rating'
        {where_clause}
        GROUP BY s.set_num, s.name, s.year, t.name, s.num_parts, s.img_url
        """
        
        if search_request.min_rating is not None:
            query += f" HAVING COALESCE(AVG(ui.rating), 0) >= %s"
            all_params.append(search_request.min_rating)
        
        query += f" {order_clause} LIMIT %s OFFSET %s"
        all_params.extend([search_request.limit, search_request.offset])
        
        cursor.execute(query, all_params)
        results = cursor.fetchall()
        
        return [SetDetailsResponse(**row) for row in results]

    except Exception as e:
        logger.error(f"Error searching sets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sets/{set_num}", response_model=SetDetailsResponse)
@time_request
async def get_set_details(
    set_num: str,
    user_id: Optional[int] = Query(None),
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Get detailed information about a specific set
    """
    try:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        query = """
        SELECT s.set_num, s.name, s.year, t.name as theme_name, s.num_parts, s.img_url,
               COALESCE(AVG(ui.rating), 0) as avg_rating,
               COUNT(ui.rating) as total_ratings,
               EXISTS(
                   SELECT 1 FROM user_collections uc 
                   WHERE uc.user_id = %s AND uc.set_num = s.set_num
               ) as is_in_user_collection,
               EXISTS(
                   SELECT 1 FROM user_wishlists uw 
                   WHERE uw.user_id = %s AND uw.set_num = s.set_num
               ) as is_in_user_wishlist
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN user_interactions ui ON s.set_num = ui.set_num AND ui.interaction_type = 'rating'
        WHERE s.set_num = %s
        GROUP BY s.set_num, s.name, s.year, t.name, s.num_parts, s.img_url
        """
        
        cursor.execute(query, (user_id, user_id, set_num))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Set not found")
        
        return SetDetailsResponse(**result)

    except Exception as e:
        logger.error(f"Error fetching set details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/themes", response_model=List[Dict[str, Any]])
@time_request
async def get_themes(db: psycopg2.extensions.connection = Depends(get_db)):
    """
    Get all available themes
    """
    try:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        query = """
        SELECT t.id, t.name, t.parent_id, COUNT(s.set_num) as set_count
        FROM themes t
        LEFT JOIN sets s ON t.id = s.theme_id
        GROUP BY t.id, t.name, t.parent_id
        ORDER BY t.name
        """
        
        cursor.execute(query)
        themes = cursor.fetchall()
        
        return [dict(theme) for theme in themes]

    except Exception as e:
        logger.error(f"Error fetching themes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/collection")
@time_request
async def add_to_collection(
    user_id: int,
    set_num: str,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Add a set to user's collection
    """
    try:
        if not _user_exists(db, user_id):
            raise HTTPException(status_code=404, detail="User not found")
        
        if not _set_exists(db, set_num):
            raise HTTPException(status_code=404, detail="Set not found")

        cursor = db.cursor()
        query = """
        INSERT INTO user_collections (user_id, set_num)
        VALUES (%s, %s)
        ON CONFLICT (user_id, set_num) DO NOTHING
        """
        
        cursor.execute(query, (user_id, set_num))
        db.commit()
        
        return {"message": "Set added to collection successfully"}

    except Exception as e:
        logger.error(f"Error adding to collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/wishlist")
@time_request
async def add_to_wishlist(
    user_id: int,
    set_num: str,
    priority: int = Query(3, ge=1, le=5),
    max_price: Optional[float] = Query(None),
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Add a set to user's wishlist
    """
    try:
        if not _user_exists(db, user_id):
            raise HTTPException(status_code=404, detail="User not found")
        
        if not _set_exists(db, set_num):
            raise HTTPException(status_code=404, detail="Set not found")

        cursor = db.cursor()
        query = """
        INSERT INTO user_wishlists (user_id, set_num, priority, max_price)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, set_num) 
        DO UPDATE SET priority = EXCLUDED.priority, max_price = EXCLUDED.max_price
        """
        
        cursor.execute(query, (user_id, set_num, priority, max_price))
        db.commit()
        
        return {"message": "Set added to wishlist successfully"}

    except Exception as e:
        logger.error(f"Error adding to wishlist: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=APIMetrics)
@time_request
async def get_api_metrics():
    """
    Get API performance metrics
    """
    return performance_monitor.get_metrics()

@app.get("/health")
async def health_check(db: psycopg2.extensions.connection = Depends(get_db)):
    """
    Health check endpoint
    """
    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation_engine": "active" if recommendation_engine else "inactive"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")
    
# Natural Language Search Endpoint
@app.post("/search/natural", response_model=NLSearchResponse)
@time_request
async def natural_language_search(
    nl_query: NaturalLanguageQuery,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Search for LEGO sets using natural language queries.
    
    Examples:
    - "I want a space station with lots of detail"
    - "Gift for 8-year-old who loves cars"
    - "Challenging build under $200"
    - "Something like the Millennium Falcon but smaller"
    """
    try:
        if not nl_recommender:
            raise HTTPException(
                status_code=503, 
                detail="Natural language processor not initialized"
            )
        
        # Process the query
        nl_result = nl_recommender.process_nl_query(nl_query.query, user_context=None)
        
        # Perform semantic search
        search_results = nl_recommender.semantic_search(
            nl_query.query, 
            top_k=nl_query.top_k
        )
        
        # Get additional details for each result
        enhanced_results = []
        for result in search_results:
            # Get user-specific data if user_id provided
            user_data = {}
            if nl_query.user_id:
                cursor = db.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT 
                        EXISTS(SELECT 1 FROM user_collections WHERE user_id = %s AND set_num = %s) as owned,
                        EXISTS(SELECT 1 FROM user_wishlists WHERE user_id = %s AND set_num = %s) as wishlisted,
                        (SELECT rating FROM user_interactions WHERE user_id = %s AND set_num = %s AND interaction_type = 'rating') as user_rating
                """, (nl_query.user_id, result['set_num'], nl_query.user_id, result['set_num'], nl_query.user_id, result['set_num']))
                user_data = cursor.fetchone() or {}
            
            # Generate match reasons
            match_reasons = _generate_match_reasons(result, nl_result.filters)
            
            enhanced_results.append(NLSearchResult(
                set_num=result['set_num'],
                name=result['name'],
                theme=result['theme'],
                year=result['year'],
                num_parts=result['num_parts'],
                relevance_score=result.get('relevance_score', 0.0),
                match_reasons=match_reasons,
                description=result['description']
            ))
        
        # Generate explanation if requested
        explanation = None
        if nl_query.include_explanation and enhanced_results:
            explanation = nl_recommender.generate_recommendation_explanation(
                nl_query.query,
                [r.dict() for r in enhanced_results[:3]]
            )
        
        return NLSearchResponse(
            query=nl_query.query,
            intent=nl_result.intent,
            extracted_filters=nl_result.filters,
            results=enhanced_results,
            explanation=explanation,
            query_understanding={
                'intent': nl_result.intent,
                'confidence': nl_result.confidence,
                'entities': nl_result.extracted_entities
            }
        )
        
    except Exception as e:
        logger.error(f"Natural language search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Semantic Similar Sets Endpoint
@app.post("/sets/similar/semantic", response_model=List[NLSearchResult])
@time_request
async def find_semantically_similar_sets(
    query: SimilarSetQuery,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Find sets semantically similar to a given set, with optional natural language context.
    
    Example: Find sets similar to 75192-1 but "easier to build and more affordable"
    """
    try:
        if not nl_recommender:
            raise HTTPException(
                status_code=503,
                detail="Natural language processor not initialized"
            )
        
        # Get the target set details
        cursor = db.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT s.*, t.name as theme_name 
            FROM sets s 
            LEFT JOIN themes t ON s.theme_id = t.id 
            WHERE s.set_num = %s
        """, (query.set_num,))
        
        target_set = cursor.fetchone()
        if not target_set:
            raise HTTPException(status_code=404, detail="Set not found")
        
        # Create search query combining set info and user requirements
        search_query = f"Sets similar to {target_set['name']} from {target_set['theme_name']} theme"
        if query.description:
            search_query += f" but {query.description}"
        
        # Perform semantic search
        results = nl_recommender.semantic_search(search_query, top_k=query.top_k + 1)
        
        # Filter out the target set itself
        results = [r for r in results if r['set_num'] != query.set_num][:query.top_k]
        
        # Convert to response format
        enhanced_results = []
        for result in results:
            match_reasons = [
                f"Semantically similar to {target_set['name']}",
                f"Relevance score: {result.get('relevance_score', 0):.2f}"
            ]
            
            if query.description:
                match_reasons.append(f"Matches requirement: {query.description}")
            
            enhanced_results.append(NLSearchResult(
                set_num=result['set_num'],
                name=result['name'],
                theme=result['theme'],
                year=result['year'],
                num_parts=result['num_parts'],
                relevance_score=result.get('relevance_score', 0.0),
                match_reasons=match_reasons,
                description=result['description']
            ))
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Semantic similarity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Conversational Recommendation Endpoint
@app.post("/recommendations/conversational", response_model=Dict[str, Any])
@time_request
async def conversational_recommendations(
    conv_query: ConversationalQuery,
    db: psycopg2.extensions.connection = Depends(get_db)
):
    """
    Handle conversational recommendation queries with context awareness.
    
    Supports multi-turn conversations about LEGO recommendations.
    """
    try:
        if not nl_recommender:
            raise HTTPException(
                status_code=503,
                detail="Natural language processor not initialized"
            )
        
        # Build context from conversation history
        context = _build_conversation_context(conv_query.conversation_history)
        context.update(conv_query.context)
        
        # Process the current query with context
        nl_result = nl_recommender.process_nl_query(
            conv_query.query,
            user_context=context
        )
        
        # Determine appropriate response based on intent
        response = {}
        
        if nl_result.intent == 'gift_recommendation':
            # Handle gift recommendation flow
            response = _handle_gift_recommendation(
                nl_recommender, nl_result, conv_query, db
            )
        elif nl_result.intent == 'collection_advice':
            # Handle collection advice flow
            response = _handle_collection_advice(
                nl_recommender, nl_result, conv_query, db
            )
        else:
            # Default search flow
            results = nl_recommender.semantic_search(
                conv_query.query,
                top_k=5,
                filters=nl_result.filters
            )
            response = {
                'type': 'recommendations',
                'results': results,
                'follow_up_questions': _generate_follow_up_questions(nl_result, results)
            }
        
        # Add query understanding to response
        response['query_understanding'] = {
            'intent': nl_result.intent,
            'extracted_info': nl_result.extracted_entities,
            'filters': nl_result.filters
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Conversational recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query Understanding Endpoint
@app.post("/nlp/understand", response_model=Dict[str, Any])
@time_request
async def understand_query(query: str = Body(..., embed=True)):
    """
    Analyze and understand a natural language query without performing search.
    
    Useful for debugging and understanding how the system interprets queries.
    """
    try:
        if not nl_recommender:
            raise HTTPException(
                status_code=503,
                detail="Natural language processor not initialized"
            )
        
        # Process query
        nl_result = nl_recommender.process_nl_query(query, None)
        
        return {
            'original_query': query,
            'intent': nl_result.intent,
            'confidence': nl_result.confidence,
            'extracted_filters': nl_result.filters,
            'extracted_entities': nl_result.extracted_entities,
            'semantic_query': nl_result.semantic_query,
            'interpretation': _generate_query_interpretation(nl_result)
        }
        
    except Exception as e:
        logger.error(f"Query understanding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _get_cached_recommendations(db, request: RecommendationRequest) -> Optional[List[RecommendationResponse]]:
    """Get cached recommendations if available and not expired"""
    if not request.user_id:
        return None
    
    try:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        query = """
        SELECT ur.set_num, s.name, ur.score, ur.reasons, t.name as theme_name,
               s.year, s.num_parts, s.img_url
        FROM user_recommendations ur
        JOIN sets s ON ur.set_num = s.set_num
        LEFT JOIN themes t ON s.theme_id = t.id
        WHERE ur.user_id = %s AND ur.recommendation_type = %s 
              AND ur.expires_at > CURRENT_TIMESTAMP
        ORDER BY ur.score DESC
        LIMIT %s
        """
        
        cursor.execute(query, (request.user_id, request.recommendation_type, request.top_k))
        results = cursor.fetchall()
        
        if results:
            return [
                RecommendationResponse(
                    set_num=row['set_num'],
                    name=row['name'],
                    score=float(row['score']),
                    reasons=row['reasons'] or [],
                    theme_name=row['theme_name'],
                    year=row['year'],
                    num_parts=row['num_parts'],
                    img_url=row['img_url']
                )
                for row in results
            ]
    except Exception as e:
        logger.error(f"Error fetching cached recommendations: {e}")
    
    return None

def _cache_recommendations(db, request: RecommendationRequest, recommendations: List[RecommendationResponse]):
    """Cache recommendations for future use"""
    if not request.user_id:
        return
    
    try:
        cursor = db.cursor()
        
        # Clear existing cache for this user and type
        cursor.execute(
            "DELETE FROM user_recommendations WHERE user_id = %s AND recommendation_type = %s",
            (request.user_id, request.recommendation_type)
        )
        
        # Insert new recommendations
        for rec in recommendations:
            cursor.execute("""
                INSERT INTO user_recommendations 
                (user_id, set_num, recommendation_type, score, reasons)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                request.user_id,
                rec.set_num,
                request.recommendation_type,
                rec.score,
                rec.reasons
            ))
        
        db.commit()
    except Exception as e:
        logger.error(f"Error caching recommendations: {e}")
        db.rollback()

def _user_exists(db, user_id: int) -> bool:
    """Check if user exists"""
    cursor = db.cursor()
    cursor.execute("SELECT 1 FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone() is not None

def _set_exists(db, set_num: str) -> bool:
    """Check if set exists"""
    cursor = db.cursor()
    cursor.execute("SELECT 1 FROM sets WHERE set_num = %s", (set_num,))
    return cursor.fetchone() is not None

async def _update_user_rating_stats(user_id: int, dsn: str):
    """Background task to update user rating statistics"""
    try:
        with psycopg2.connect(dsn) as conn:
            cursor = conn.cursor()
            query = """
            UPDATE users SET
                total_ratings = (
                    SELECT COUNT(*) FROM user_interactions 
                    WHERE user_id = %s AND interaction_type = 'rating'
                ),
                avg_rating = (
                    SELECT COALESCE(AVG(rating), 0) FROM user_interactions 
                    WHERE user_id = %s AND interaction_type = 'rating'
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """
            cursor.execute(query, (user_id, user_id, user_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Error updating user rating stats: {e}")

if __name__ == "__main__":
    import uvicorn
    print("✅ FastAPI app initialized successfully! Starting uvicorn server...")
    print("🌐 Server will be available at http://0.0.0.0:8000")
    print("📖 API documentation: http://0.0.0.0:8000/docs")
    print("🔄 Health check: http://0.0.0.0:8000/health")
    uvicorn.run(
        "recommendation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to prevent interruption during startup
        log_level="info",
        access_log=True
    )

def _generate_match_reasons(result: Dict, filters: Dict) -> List[str]:
    """Generate human-readable match reasons"""
    reasons = []
    
    # Theme match
    if filters.get('themes') and any(t.lower() in result['theme'].lower() for t in filters['themes']):
        reasons.append(f"Matches requested theme: {result['theme']}")
    
    # Piece count match
    if filters.get('min_pieces') and filters.get('max_pieces'):
        if filters['min_pieces'] <= result['num_parts'] <= filters['max_pieces']:
            reasons.append(f"Piece count ({result['num_parts']}) in requested range")
    
    # Complexity match
    if filters.get('complexity'):
        reasons.append(f"Matches {filters['complexity']} complexity preference")
    
    # High relevance score
    if result.get('relevance_score', 0) > 0.8:
        reasons.append("High semantic similarity to your query")
    
    return reasons if reasons else ["Matches your search criteria"]

def _build_conversation_context(history: List[Dict[str, str]]) -> Dict:
    """Build context from conversation history"""
    context = {
        'mentioned_themes': set(),
        'mentioned_sets': set(),
        'price_range': None,
        'age_range': None
    }
    
    # Extract information from history
    for turn in history:
        if turn.get('role') == 'user':
            # Extract themes, sets, prices, etc. from previous queries
            # This would be more sophisticated in production
            pass
    
    return context

def _generate_follow_up_questions(nl_result: NLQueryResult, results: List[Dict]) -> List[str]:
    """Generate relevant follow-up questions"""
    questions = []
    
    if not nl_result.filters.get('budget_min'):
        questions.append("What's your budget for this set?")
    
    if not nl_result.filters.get('themes'):
        themes = list(set(r['theme'] for r in results[:3]))
        if themes:
            questions.append(f"Are you particularly interested in any of these themes: {', '.join(themes)}?")
    
    if nl_result.intent == 'gift_recommendation' and not nl_result.extracted_entities.get('recipient'):
        questions.append("Who is this gift for? (age and interests would help)")
    
    return questions

def _generate_query_interpretation(nl_result: NLQueryResult) -> str:
    """Generate human-readable interpretation of query understanding"""
    parts = []
    
    if nl_result.intent == 'search':
        parts.append("I understand you're searching for LEGO sets")
    elif nl_result.intent == 'gift_recommendation':
        parts.append("I see you're looking for a LEGO gift")
    elif nl_result.intent == 'recommend_similar':
        parts.append("You want sets similar to one you know")
    
    if nl_result.filters:
        filter_desc = []
        if nl_result.filters.get('themes'):
            filter_desc.append(f"from {', '.join(nl_result.filters['themes'])} themes")
        if nl_result.filters.get('min_pieces') or nl_result.filters.get('max_pieces'):
            if nl_result.filters.get('min_pieces') and nl_result.filters.get('max_pieces'):
                filter_desc.append(f"with {nl_result.filters['min_pieces']}-{nl_result.filters['max_pieces']} pieces")
            elif nl_result.filters.get('min_pieces'):
                filter_desc.append(f"with at least {nl_result.filters['min_pieces']} pieces")
            else:
                filter_desc.append(f"with up to {nl_result.filters['max_pieces']} pieces")
        
        if filter_desc:
            parts.append("with these criteria: " + ", ".join(filter_desc))
    
    return " ".join(parts) + "."

def _handle_gift_recommendation(nl_recommender, nl_result, conv_query, db):
    """Handle gift recommendation intent"""
    # Would implement gift-specific logic
    # For now, return standard recommendations with gift context
    results = nl_recommender.semantic_search(
        conv_query.query,
        top_k=5,
        filters=nl_result.filters
    )
    
    return {
        'type': 'gift_recommendations',
        'results': results,
        'gift_context': nl_result.extracted_entities,
        'follow_up_questions': [
            "What's your budget for this gift?",
            "Does the recipient have any favorite themes or interests?",
            "Are they new to LEGO or an experienced builder?"
        ]
    }

def _handle_collection_advice(nl_recommender, nl_result, conv_query, db):
    """Handle collection advice intent"""
    # Would implement collection-specific logic
    # For now, return standard recommendations with collection context
    results = nl_recommender.semantic_search(
        conv_query.query,
        top_k=5,
        filters=nl_result.filters
    )
    
    return {
        'type': 'collection_advice',
        'results': results,
        'advice': "These sets would make great additions to your collection based on current market trends and theme popularity.",
        'follow_up_questions': [
            "What themes do you currently collect?",
            "Are you looking for investment pieces or personal enjoyment?",
            "Do you prefer sealed sets or are you planning to build them?"
        ]
    }
