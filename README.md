# 🧱 LEGO Recommendation System

A production-ready, sophisticated recommendation system that helps LEGO enthusiasts discover sets they'll love based on their preferences, building history, and advanced ML algorithms.

## 🎯 Features

- **🤖 Hybrid ML Engine**: Content-based + Collaborative filtering + Smart fallbacks
- **🧠 Advanced NLP Processing**: LangChain-powered natural language understanding
- **🗣️ Conversational AI**: Multi-turn dialogue with context awareness
- **🔍 Semantic Search**: FAISS vector database with HuggingFace embeddings
- **🎯 Intent Detection**: Understands gift recommendations, similar sets, collection advice
- **📝 Query Understanding**: Extracts filters, entities, and semantic meaning from natural language
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
│   ├── recommendation_api.py      # FastAPI service with NL endpoints
│   ├── recommendation_system.py   # ML recommendation engine
│   ├── lego_nlp_recommeder.py     # LangChain NLP processor
│   └── upload_rebrickable_data.py # Data loading utilities
├── db/
│   ├── rebrickable_schema.sql     # LEGO sets database schema
│   └── user_interaction_schema.sql # User data schema
tests/
├── unit/                          # Component-level tests
├── integration/                   # End-to-end API tests including NL
└── performance/                   # Load and scalability tests
examples/
├── example_client.py              # Complete API demonstration
└── nl_demo_script.py              # Natural language demo
data/
└── rebrickable/                   # LEGO dataset (CSV files)
embeddings/                        # Vector database storage
```

## 🚀 Quick Start

### Method 1: Complete Setup and Start (Recommended)
```bash
# One command to set up everything and start the system
./scripts/quick_setup.sh

# This script:
# - Resets and sets up the database
# - Starts PostgreSQL and FastAPI containers
# - Sets up conda environment with all dependencies
# - Installs natural language processing features
# - Loads LEGO data (if available)
# - Starts the API server on http://localhost:8000
```

### Method 2: Next Manual Setup (Step by Step)
```bash
### 1. Load LEGO data, Start FastAPI, and Install Dependencies
./scripts/quick_setup.sh

### 2. Set Ollama LLM (if using local inference)
# Ensure Ollama is running and accessible
```bash
./scripts/setup_ollama.sh
./scripts/setup_ollama_models.sh
```

### 3. Test the System
```bash
# Run all tests (basic) - scripts are now in /scripts folder
./scripts/run_all_tests.sh

# Run all tests including optional ones
./scripts/run_all_tests.sh --all

# Run specific test categories
./scripts/run_all_tests.sh --integration --performance --nl-advanced
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
- `POST /search/natural` - **Natural language search** ("star wars sets for kids")
- `POST /nlp/understand` - **Query understanding** and intent detection
- `POST /recommendations/conversational` - **Multi-turn conversations** with context
- `POST /sets/similar/semantic` - **Semantic similarity** search with descriptions
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
./scripts/run_all_tests.sh

# Comprehensive testing (includes all optional tests)
./scripts/run_all_tests.sh --all

# Specific test categories
./scripts/run_all_tests.sh --integration    # End-to-end workflows
./scripts/run_all_tests.sh --performance    # Load and speed testing
./scripts/run_all_tests.sh --nl-advanced    # Advanced NL features
./scripts/run_all_tests.sh --examples       # Example scripts

# Combined categories
./run_all_tests.sh --integration --performance
```

### Test Categories
- **Unit Tests**: Database connectivity, recommendation algorithms, NLP components
- **Integration Tests**: End-to-end API workflows, NL processing, conversational AI
- **Performance Tests**: Load testing, response time validation, semantic search speed
- **API Tests**: Endpoint availability, error handling, natural language endpoints
- **Natural Language Tests**: Query understanding, semantic search, intent detection, filter extraction
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

### 5. **Natural Language Processing**
- **Intent Classification**: Gift recommendations, similar sets, collection advice, general search
- **Filter Extraction**: Automatically parse piece counts, themes, complexity, age ranges
- **Entity Recognition**: Identify recipients, occasions, preferences from queries
- **Semantic Search**: FAISS vector database with HuggingFace embeddings
- **Conversational Context**: Multi-turn dialogue with memory and follow-up questions

### 6. **AI-Powered Understanding**
- **LangChain Integration**: Modern prompt engineering and LLM chains
- **Ollama Support**: Local LLM inference for privacy-conscious deployments
- **Confidence Scoring**: Dynamic confidence based on query clarity and extracted information
- **Explanation Generation**: Human-readable explanations for all recommendations

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

### Core Infrastructure
- **Backend**: FastAPI with automatic documentation and validation
- **Database**: PostgreSQL 15+ with optimized schema design
- **Containerization**: Docker Compose with multi-service orchestration
- **Environment Management**: Conda with reproducible environment specifications
- **Process Management**: Multi-threaded recommendation processing

### Machine Learning & AI
- **NLP Framework**: LangChain for prompt engineering and LLM integration
- **Vector Database**: FAISS for high-performance semantic similarity search
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Local LLM**: Ollama support for privacy-conscious deployments
- **Traditional ML**: Scikit-learn, Pandas, NumPy for numerical processing
- **Fallback Processing**: Graceful degradation when AI services unavailable

### Web Framework & API
- **REST API**: FastAPI with automatic OpenAPI documentation
- **API Design**: RESTful endpoints with comprehensive error handling
- **Content Types**: JSON responses with structured error messages
- **CORS Support**: Configurable cross-origin resource sharing
- **Performance**: Sub-second response times with caching

### Data Processing
- **Data Source**: Rebrickable.com LEGO database (700k+ parts, 20k+ sets)
- **Schema Design**: Optimized joins and indexing for recommendation queries
- **Caching**: In-memory theme and category caching for performance
- **Testing**: Comprehensive unit/integration/performance test coverage
- **Deployment**: Production-ready containerized setup with health checks

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

### Natural Language Search
```python
# Natural language queries with AI understanding
response = requests.post("http://localhost:8000/search/natural", json={
    "query": "Star Wars sets under 500 pieces for my 8-year-old nephew's birthday",
    "top_k": 10
})

# Query understanding and intent detection
response = requests.post("http://localhost:8000/nlp/understand", json={
    "query": "I need something similar to the Hogwarts Castle but smaller and less expensive"
})

# Conversational AI recommendations
response = requests.post("http://localhost:8000/recommendations/conversational", json={
    "query": "What's a good gift for someone who loves medieval themes?",
    "conversation_id": "user123_session1"
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

# Test natural language processing
curl -X POST http://localhost:8000/search/natural \
  -H "Content-Type: application/json" \
  -d '{"query": "space sets under 200 pieces", "top_k": 5}'

# Test conversational AI
curl -X POST http://localhost:8000/recommendations/conversational \
  -H "Content-Type: application/json" \
  -d '{"query": "What would you recommend for a Batman fan?", "conversation_id": "test123"}'

# Run the example client from container
docker exec brickbrain-app conda run -n brickbrain-rec python /app/examples/example_client.py

# Run the NLP demo script
docker exec brickbrain-app conda run -n brickbrain-rec python /app/examples/nl_demo_script.py
```

## 🏆 System Status

**🎉 PRODUCTION READY WITH ADVANCED NLP!**

- Core recommendation engine: ✅ 100% functional
- Natural language processing: ✅ LangChain + Ollama integration
- Semantic search: ✅ FAISS vector database with HuggingFace embeddings
- Conversational AI: ✅ Multi-turn dialogue with context awareness
- API service: ✅ Fully operational with NL endpoints
- Database: ✅ Optimized and loaded with theme caching
- Testing: ✅ Comprehensive coverage (18/18 tests passing)
- Performance: ✅ Sub-second response times with vector search
- Scalability: ✅ Load tested and production validated

The system successfully handles:
- ✅ Cold start scenarios (new users)
- ✅ Content-based recommendations with semantic similarity
- ✅ Collaborative filtering with user behavior analysis
- ✅ Hybrid intelligent recommendations combining multiple signals
- ✅ Natural language search with intent understanding
- ✅ Conversational AI with context and follow-up questions
- ✅ Filter extraction from complex queries ("between X and Y pieces")
- ✅ Entity recognition (recipients, occasions, preferences)
- ✅ Search and discovery with semantic matching
- ✅ User management and interaction tracking
- ✅ Production-level performance with vector optimization

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

**NLP/AI features not working:**
```bash
# Check LangChain installation
docker exec brickbrain-app conda run -n brickbrain-rec python -c "import langchain; print('LangChain:', langchain.__version__)"

# Check FAISS vector database
docker exec brickbrain-app conda run -n brickbrain-rec python -c "import faiss; print('FAISS:', faiss.__version__)"

# Check HuggingFace transformers
docker exec brickbrain-app conda run -n brickbrain-rec python -c "from transformers import AutoModel; print('Transformers: OK')"

# Test Ollama connection (if using local LLM)
curl http://localhost:11434/api/tags  # Check if Ollama is running

# Run NLP integration tests
docker exec brickbrain-app conda run -n brickbrain-rec python /app/tests/integration/nl_integration_test.py
```

**Vector embeddings issues:**
```bash
# Check embeddings directory
docker exec brickbrain-app ls -la /app/embeddings/

# Regenerate embeddings if corrupted
docker exec brickbrain-app conda run -n brickbrain-rec python -c "
from src.scripts.lego_nlp_recommeder import LEGONLPRecommender
recommender = LEGONLPRecommender()
# This will recreate embeddings if missing
"
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
- ✅ Installs natural language processing packages (LangChain, FAISS, transformers)
- ✅ Downloads NLTK and spaCy models for text processing
- ✅ Loads LEGO data (if available in data/rebrickable/)
- ✅ Initializes FAISS vector database for semantic search
- ✅ Downloads HuggingFace embeddings model (all-MiniLM-L6-v2)
- ✅ Starts FastAPI server with NLP endpoints on http://localhost:8000
- ✅ Runs basic functionality and NLP integration tests

**Perfect for:**
- First-time setup
- Fresh development environment
- Resetting after major changes
- Demonstration or deployment

### 2. `./scripts/run_all_tests.sh` - Comprehensive Testing
**Flexible test execution:**
```bash
# Basic tests (always run)
./scripts/run_all_tests.sh

# All tests including optional ones
./scripts/run_all_tests.sh --all

# Specific categories
./scripts/run_all_tests.sh --integration --performance --nl-advanced --examples
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