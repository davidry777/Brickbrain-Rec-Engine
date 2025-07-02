# 🧱 LEGO Recommendation System

A production-ready, sophisticated recommendation system that helps LEGO enthusiasts discover sets they'll love based on their preferences, building history, and advanced ML algorithms.

## 🎯 Features

- **🤖 Hybrid ML Engine**: Content-based + Collaborative filtering + Smart fallbacks
- **🚀 FastAPI Service**: Production-ready REST API with comprehensive endpoints
- **📊 Rich Database**: 25,216+ LEGO sets with complete metadata across 479 themes
- **🔍 Advanced Search**: Filter by theme, complexity, year, pieces, and more
- **👥 User Management**: Profile creation, preferences, and interaction tracking
- **📈 Analytics**: Performance metrics, recommendation quality tracking
- **⚡ High Performance**: <1s response time, 20+ concurrent requests/second

## 🏗️ Architecture

```
src/
├── scripts/
│   ├── recommendation_api.py      # FastAPI service
│   ├── recommendation_system.py   # ML recommendation engine
│   └── upload_rebrickable_data.py # Data loading utilities
├── db/
│   ├── rebrickable_schema.sql     # LEGO sets database schema
│   └── user_interaction_schema.sql # User data schema
tests/
├── unit/                          # Component-level tests
├── integration/                   # End-to-end API tests  
└── performance/                   # Load and scalability tests
examples/
└── example_client.py              # Complete API demonstration
data/
└── rebrickable/                   # LEGO dataset (CSV files)
```

## 🚀 Quick Start

### 1. Setup Database
```bash
# Start PostgreSQL
docker-compose up -d

# Load LEGO data (25K+ sets)
./reset_db.sh
```

### 2. Start API Service
```bash
python src/scripts/recommendation_api.py
```

### 3. Test the System
```bash
# Quick validation
./run_tests.sh

# Or run specific tests
python tests/integration/production_test_simple.py
```

### 4. Try the Demo
```bash
python examples/example_client.py
```

## 📋 API Endpoints

### Core Recommendations
- `POST /recommendations` - Get personalized recommendations
- `GET /health` - System health check
- `GET /metrics` - Performance analytics

### User Management  
- `POST /users` - Create user account
- `GET /users/{user_id}/profile` - Get user profile
- `POST /users/{user_id}/interactions` - Record ratings/interactions

### Search & Discovery
- `POST /search/sets` - Search LEGO sets with filters
- `GET /themes` - List all available themes
- `GET /sets/{set_num}` - Get detailed set information

### Collections
- `POST /users/{user_id}/wishlist` - Manage wishlist
- `POST /users/{user_id}/collection` - Track owned sets

## 🧪 Testing

The system includes comprehensive test coverage:

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh --performance  # Include load testing
./run_tests.sh --all          # Everything including examples

# Individual test suites
python tests/unit/test_database.py           # Database connectivity
python tests/unit/test_recommendations.py    # ML algorithms
python tests/integration/final_validation.py # Complete API validation
python tests/performance/production_scalability_test.py # Load testing
```

## 📊 Recommendation Types

### 1. **Content-Based** 
- Analyzes set features (theme, pieces, complexity, year)
- Works immediately for any set
- Perfect for "sets similar to this one"

### 2. **Collaborative Filtering**
- Uses user rating patterns and preferences  
- Improves with more user data
- Great for discovering new themes

### 3. **Hybrid Approach**
- Intelligently combines both methods
- Includes smart fallbacks for edge cases
- Optimizes for best recommendation quality

### 4. **Cold Start Handling**
- Popular/trending sets for new users
- Theme-based recommendations from preferences
- Graceful degradation when data is sparse

## 🎯 Production Ready

✅ **Validated & Tested**
- 100% core functionality working
- Load tested with 1000+ simulated users
- Comprehensive error handling and fallbacks
- Performance optimized (<1s response time)

✅ **Scalable Architecture**
- Stateless API design
- Database connection pooling
- Efficient caching strategies
- Horizontal scaling ready

✅ **Monitoring & Analytics**
- Request/response time tracking
- Recommendation quality metrics
- User engagement analytics
- System health monitoring

## 📈 Performance Metrics

- **Response Time**: <1 second average
- **Throughput**: 20+ requests/second
- **Recommendation Quality**: 75-95% confidence scores
- **Theme Diversity**: 2-5 themes per recommendation set
- **Database**: 25,216 sets, 479 themes, instant queries

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Database**: PostgreSQL with optimized schemas
- **ML/AI**: Scikit-learn, Pandas, NumPy
- **Testing**: Comprehensive unit/integration/performance tests
- **Data**: Official Rebrickable LEGO database
- **Deployment**: Docker, environment-based configuration

## 📝 Configuration

Environment variables:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=brickbrain
DB_USER=brickbrain
DB_PASSWORD=brickbrain_password
```

## 🤝 API Usage Examples

### Get Recommendations
```python
import requests

# Content-based (similar sets)
response = requests.post("http://localhost:8000/recommendations", json={
    "recommendation_type": "content",
    "set_num": "75192-1",  # Millennium Falcon
    "top_k": 5
})

# User-based (personalized)
response = requests.post("http://localhost:8000/recommendations", json={
    "recommendation_type": "collaborative", 
    "user_id": 123,
    "top_k": 10
})

# Hybrid (best of both)
response = requests.post("http://localhost:8000/recommendations", json={
    "recommendation_type": "hybrid",
    "user_id": 123,
    "set_num": "10242-1", 
    "top_k": 10
})
```

### Search Sets
```python
response = requests.post("http://localhost:8000/search/sets", json={
    "query": "star wars",
    "min_pieces": 100,
    "max_pieces": 1000,
    "theme_ids": [158],  # Star Wars theme
    "min_year": 2020,
    "top_k": 20
})
```

## 🏆 System Status

**🎉 PRODUCTION READY!**

- Core recommendation engine: ✅ 100% functional
- API service: ✅ Fully operational
- Database: ✅ Optimized and loaded
- Testing: ✅ Comprehensive coverage
- Performance: ✅ Sub-second response times
- Scalability: ✅ Load tested and validated

The system successfully handles:
- ✅ Cold start scenarios (new users)
- ✅ Content-based recommendations
- ✅ Collaborative filtering
- ✅ Hybrid intelligent recommendations  
- ✅ Search and discovery
- ✅ User management and tracking
- ✅ Production-level performance

## 📚 Documentation

- [Test Suite Documentation](tests/README.md)
- [API Examples](examples/README.md)
- [Production Readiness Summary](PRODUCTION_READY_SUMMARY.md)

---

**Built with ❤️ for LEGO enthusiasts everywhere**