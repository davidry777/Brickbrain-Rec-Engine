# 🧱 LEGO Recommendation Engine API

A production-ready FastAPI service that provides personalized LEGO set recommendations using machine learning algorithms.

## 🚀 Quick Start

### 1. Start the API Server
```bash
./start_api.sh
```

### 2. Access the API
- **API Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📋 API Endpoints

### 🎯 Core Recommendation Endpoints

#### Get Recommendations
```http
POST /recommendations
```
Get personalized recommendations for a user or find similar sets.

**Request Body:**
```json
{
  "user_id": 1,
  "set_num": "75192-1",
  "top_k": 10,
  "recommendation_type": "hybrid",
  "include_reasons": true
}
```

**Response:**
```json
[
  {
    "set_num": "75059-1",
    "name": "Sandcrawler",
    "score": 0.95,
    "reasons": ["Same theme: Star Wars", "Similar complexity level"],
    "theme_name": "Star Wars",
    "year": 2014,
    "num_parts": 3296,
    "img_url": "https://cdn.rebrickable.com/media/sets/75059-1/12345.jpg"
  }
]
```

### 👤 User Management

#### Create User
```http
POST /users
```

#### Get User Profile
```http
GET /users/{user_id}/profile
```

#### Update User Preferences
```http
PUT /users/{user_id}/preferences
```

### 📊 User Interactions

#### Track User Interaction
```http
POST /users/{user_id}/interactions
```
Track ratings, views, purchases, wishlist additions, and clicks.

**Request Body:**
```json
{
  "user_id": 1,
  "set_num": "75192-1",
  "interaction_type": "rating",
  "rating": 5,
  "source": "recommendation",
  "session_id": "abc123"
}
```

**Interaction Types:**
- `rating` - User rated a set (1-5 stars)
- `view` - User viewed set details
- `purchase` - User purchased the set
- `wishlist` - User added to wishlist
- `click` - User clicked on a recommendation

### 🔍 Search & Discovery

#### Search Sets
```http
POST /search/sets
```

**Request Body:**
```json
{
  "query": "star wars",
  "theme_ids": [158],
  "min_pieces": 100,
  "max_pieces": 1000,
  "min_year": 2020,
  "max_year": 2024,
  "min_rating": 4.0,
  "sort_by": "name",
  "sort_order": "asc",
  "limit": 20,
  "offset": 0
}
```

#### Get Set Details
```http
GET /sets/{set_num}?user_id={user_id}
```

#### Get All Themes
```http
GET /themes
```

### 📚 User Collections

#### Add to Collection
```http
POST /users/{user_id}/collection?set_num={set_num}
```

#### Add to Wishlist
```http
POST /users/{user_id}/wishlist?set_num={set_num}&priority=3&max_price=299.99
```

### 📈 Monitoring

#### API Metrics
```http
GET /metrics
```

#### Health Check
```http
GET /health
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lego_recommendations
DB_USER=postgres
DB_PASSWORD=password
```

### Database Setup

1. **Ensure PostgreSQL is running**
2. **Create the database:**
   ```sql
   CREATE DATABASE lego_recommendations;
   ```

3. **Load the schemas:**
   ```bash
   psql -d lego_recommendations -f src/db/rebrickable_schema.sql
   psql -d lego_recommendations -f src/db/user_interaction_schema.sql
   ```

4. **Load sample data:**
   ```bash
   python src/scripts/upload_rebrickable_data.py
   ```

## 🏗️ Architecture

```
Frontend App → FastAPI Service → Recommendation Engine → PostgreSQL Database
    ↓              ↓                    ↓                    ↓
  User clicks   REST API          ML Algorithms        LEGO set data
  Web/Mobile    JSON responses    Content/Collab       User interactions
```

### Key Components

1. **FastAPI Server** (`recommendation_api.py`)
   - RESTful API endpoints
   - Request validation with Pydantic
   - Database connection management
   - Performance monitoring
   - Caching system

2. **Recommendation Engine** (`recommendation_system.py`)
   - Content-based filtering
   - Collaborative filtering
   - Hybrid recommendations
   - ML model training

3. **Database Layer**
   - PostgreSQL with optimized schemas
   - User interaction tracking
   - Recommendation caching
   - Performance indexes

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
EXPOSE 8000

CMD ["uvicorn", "src.scripts.recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
See `docker-compose.yml` for full stack deployment.

### Performance Optimization

1. **Caching**: Recommendations cached for 24 hours
2. **Database Indexing**: Optimized queries for user interactions
3. **Connection Pooling**: Efficient database connection management
4. **Background Tasks**: Async processing for statistics updates

## 📊 Monitoring & Metrics

The API provides built-in monitoring:

- **Request Count**: Total API requests served
- **Response Time**: Average response time tracking
- **Cache Hit Rate**: Recommendation cache efficiency
- **User Activity**: Active users and interactions

## 🧪 Testing

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "top_k": 5,
    "recommendation_type": "hybrid"
  }'

# Search sets
curl -X POST "http://localhost:8000/search/sets" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "star wars",
    "limit": 10
  }'
```

### Interactive Testing

Visit http://localhost:8000/docs for the interactive Swagger UI where you can test all endpoints directly in your browser.

## 🔒 Security Considerations

For production deployment:

1. **Authentication**: Implement JWT or OAuth2
2. **Rate Limiting**: Add rate limiting middleware
3. **CORS**: Configure proper CORS origins
4. **HTTPS**: Use TLS encryption
5. **Input Validation**: Enhanced input sanitization
6. **Database Security**: Use connection pooling and prepared statements

## 📈 Scaling Considerations

- **Horizontal Scaling**: Run multiple API instances behind a load balancer
- **Database Optimization**: Read replicas for recommendation queries
- **Caching Layer**: Redis for distributed caching
- **CDN**: For static assets and images
- **Monitoring**: Application Performance Monitoring (APM) tools

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check PostgreSQL is running
   - Verify connection parameters
   - Ensure database exists

2. **Recommendation Engine Not Initialized**
   - Check database has required data
   - Verify schemas are loaded
   - Review logs for initialization errors

3. **Performance Issues**
   - Check database indexes
   - Monitor cache hit rates
   - Review query performance

### Logs

API logs are written to stdout and can be viewed with:
```bash
# If running with the script
tail -f logs/api.log

# If running with Docker
docker logs container-name
```
