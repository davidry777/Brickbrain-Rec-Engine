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

### Method 1: Complete Setup and Start (Recommended)
```bash
# One command to set up everything and start the system
./setup_and_start.sh

# This script:
# - Resets and sets up the database
# - Starts PostgreSQL and FastAPI containers
# - Sets up conda environment with all dependencies
# - Installs natural language processing features
# - Loads LEGO data (if available)
# - Starts the API server on http://localhost:8000
```

### Method 2: Manual Setup (Step by Step)
```bash
# 1. Start containers
docker-compose up -d

# 2. Load LEGO data
./reset_db.sh

# 3. Set up natural language features (optional)
./setup_nl_features.sh
```

### 3. Test the System
```bash
# Run all tests (basic)
./run_all_tests.sh

# Run all tests including optional ones
./run_all_tests.sh --all

# Run specific test categories
./run_all_tests.sh --integration --performance --nl-advanced
```

### 4. Try the Demo
```bash
# Run the example client
docker exec brickbrain-app conda run -n brickbrain-rec python /app/examples/example_client.py
```

### 5. Access the API
- **API Health**: http://localhost:8000/health
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📋 API Endpoints

### Core Recommendations
- `POST /recommendations` - Get personalized recommendations
- `POST /search/natural` - **NEW**: Natural language search ("star wars sets for kids")
- `POST /nlp/understand` - **NEW**: Query understanding and intent detection
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

The system includes comprehensive test coverage with streamlined execution:

```bash
# Basic testing (unit tests, API health, core NL features)
./run_all_tests.sh

# Comprehensive testing (includes all optional tests)
./run_all_tests.sh --all

# Specific test categories
./run_all_tests.sh --integration    # End-to-end workflows
./run_all_tests.sh --performance    # Load and speed testing
./run_all_tests.sh --nl-advanced    # Advanced NL features
./run_all_tests.sh --examples       # Example scripts

# Combined categories
./run_all_tests.sh --integration --performance
```

### Test Categories
- **Unit Tests**: Database connectivity, recommendation algorithms
- **Integration Tests**: End-to-end API workflows, NL processing
- **Performance Tests**: Load testing, response time validation
- **API Tests**: Endpoint availability, error handling
- **Natural Language Tests**: Query understanding, semantic search
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/unit/test_database.py
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/unit/test_recommendations.py
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/integration/final_validation.py
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/performance/production_scalability_test.py

# Alternative: Run from host (if you have dependencies installed locally)
./run_tests.sh
```

### 🐳 Docker Commands

```bash
# Start the system
docker-compose up -d

# Check container status
docker ps

# View API logs
docker logs brickbrain-app

# Access container shell
docker exec -it brickbrain-app bash

# Stop the system
docker-compose down
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
- **Environment**: Conda with comprehensive dependency management
- **Containerization**: Docker & Docker Compose
- **Testing**: Comprehensive unit/integration/performance tests
- **Data**: Official Rebrickable LEGO database
- **Deployment**: Production-ready containerized setup

## 🐳 Docker Architecture

```yaml
services:
  postgres:     # PostgreSQL database
  app:          # Python/Conda environment with FastAPI
```

The system uses:
- **Conda Environment**: `brickbrain-rec` with all ML dependencies
- **Persistent Volumes**: Database and conda environments
- **Health Checks**: Automatic container health monitoring
- **Environment Variables**: Flexible configuration

## 📝 Configuration

The system uses Docker Compose with environment variables:

```yaml
# docker-compose.yml automatically sets:
DB_HOST=postgres           # Container name for database
DB_PORT=5432
DB_NAME=brickbrain
DB_USER=brickbrain
DB_PASSWORD=brickbrain_password
```

### Container Access
```bash
# Access the application container
docker exec -it brickbrain-app bash

# Run commands in the conda environment
docker exec brickbrain-app conda run -n brickbrain-rec python --version

# Check available packages
docker exec brickbrain-app conda list -n brickbrain-rec
```

## 🤝 API Usage Examples

The API runs in Docker and is accessible at `http://localhost:8000`:

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

### Test the API
```bash
# Quick health check
curl http://localhost:8000/health

# Run the example client from container
docker exec brickbrain-app conda run -n brickbrain-rec python /app/examples/example_client.py
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

## 🔧 Troubleshooting

### Common Issues and Solutions

**Container won't start:**
```bash
# Check if ports are in use
docker ps -a
lsof -i :5432  # PostgreSQL
lsof -i :8000  # FastAPI

# Restart the system
docker-compose down
docker-compose up -d
```

**Tests fail with "module not found":**
```bash
# Ensure you're using the container environment
docker exec brickbrain-app conda run -n brickbrain-rec python --version

# Check conda environment exists
docker exec brickbrain-app conda env list
```

**Database connection issues:**
```bash
# Check PostgreSQL container health
docker exec brickbrain-postgres pg_isready -U brickbrain

# Reset database if needed
./reset_db.sh
```

**API not responding:**
```bash
# Check API container logs
docker logs brickbrain-app

# Check API health
curl http://localhost:8000/health
```

**Performance issues:**
```bash
# Check container resources
docker stats

# Monitor API performance
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/performance/production_scalability_test.py
```

## 🛠️ Streamlined Scripts

The project includes two main scripts for easy setup and testing:

### 1. `setup_and_start.sh` - Complete System Setup
**One-command setup and startup:**
```bash
./setup_and_start.sh
```

**What it does:**
- ✅ Resets and sets up the PostgreSQL database
- ✅ Creates required directories and environment files
- ✅ Starts Docker containers (database and application)
- ✅ Sets up conda environment with all dependencies
- ✅ Installs natural language processing packages
- ✅ Downloads NLTK and spaCy models
- ✅ Loads LEGO data (if available)
- ✅ Initializes vector database for semantic search
- ✅ Starts FastAPI server on http://localhost:8000
- ✅ Runs basic functionality tests

**Perfect for:**
- First-time setup
- Fresh development environment
- Resetting after major changes
- Demonstration or deployment

### 2. `run_all_tests.sh` - Comprehensive Testing
**Flexible test execution:**
```bash
# Basic tests (always run)
./run_all_tests.sh

# All tests including optional ones
./run_all_tests.sh --all

# Specific categories
./run_all_tests.sh --integration --performance --nl-advanced --examples
```

**Test Categories:**
- **Core Tests** (always run): Unit tests, API health, basic NL features
- **Integration Tests** (`--integration`): End-to-end workflows
- **Performance Tests** (`--performance`): Load and scalability testing
- **Advanced NL Tests** (`--nl-advanced`): Comprehensive natural language validation
- **Example Scripts** (`--examples`): Usage demonstrations

**Perfect for:**
- Development testing
- CI/CD pipelines
- Quality assurance
- Performance validation

### Script Benefits
- **🚀 Fast Setup**: One command gets everything running
- **🧪 Comprehensive Testing**: All test types with flexible options
- **📊 Clear Output**: Color-coded status and detailed summaries
- **🔧 Error Handling**: Graceful failure handling and troubleshooting tips
- **📈 Progress Tracking**: Real-time status updates during setup
- **⚡ Optimized**: Efficient dependency installation and caching

### Legacy Scripts (Still Available)
- `reset_db.sh`: Database-only reset
- `setup_nl_features.sh`: NL features only setup
- `run_tests.sh`: Original test runner
- `start_api.sh`: API server only startup