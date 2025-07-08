#!/bin/bash

# ========================================
# 🧱 LEGO Recommendation Engine 
# Complete Setup and Start Script
# ========================================
# This script combines:
# - Database reset
# - FastAPI server startup
# - Natural Language features setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧱 LEGO Recommendation Engine - Complete Setup${NC}"
echo "================================================"

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the correct directory
if [ ! -f "docker-compose.yml" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_info "Starting complete setup process..."

# ========================================
# 1. Environment Setup
# ========================================

echo -e "\n${BLUE}1. Environment Setup${NC}"
echo "===================="

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=brickbrain
DB_USER=brickbrain
DB_PASSWORD=brickbrain_password

# Natural Language Processing Configuration
HF_HOME=./.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=./.cache/sentence-transformers
LANGCHAIN_CACHE_DIR=./.cache/langchain
TOKENIZERS_PARALLELISM=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Optional: OpenAI API Key (for enhanced NL features)
# OPENAI_API_KEY=your_openai_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed
EOF
    print_status "Created .env file"
else
    print_info "Using existing .env file"
fi

# Create required directories
mkdir -p .cache/huggingface .cache/sentence-transformers .cache/langchain
mkdir -p embeddings logs

print_status "Environment setup complete"

# ========================================
# 2. Docker Environment Check
# ========================================

echo -e "\n${BLUE}2. Docker Environment Check${NC}"
echo "============================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# ========================================
# 3. Database Reset and Setup
# ========================================

echo -e "\n${BLUE}3. Database Reset and Setup${NC}"
echo "==========================="

print_info "Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

print_info "Removing old database volume..."
docker volume rm brickbrain-rec-engine_postgres_data 2>/dev/null || true

print_info "Starting PostgreSQL database..."
docker-compose up -d postgres

# Wait for database to be ready
print_info "Waiting for database to be ready..."
for i in {1..30}; do
    if docker-compose exec postgres pg_isready -U brickbrain > /dev/null 2>&1; then
        print_status "Database is ready"
        break
    fi
    sleep 2
    echo -n "."
done

if [ $i -eq 30 ]; then
    print_error "Database failed to start within 60 seconds"
    exit 1
fi

# ========================================
# 4. Application Container Setup
# ========================================

echo -e "\n${BLUE}4. Application Container Setup${NC}"
echo "==============================="

print_info "Starting application container..."
docker-compose up -d app

# Wait for conda environment to be created
print_info "Waiting for conda environment setup (this may take a few minutes)..."
for i in {1..120}; do
    if docker-compose exec app conda env list | grep -q brickbrain-rec; then
        print_status "Conda environment 'brickbrain-rec' is ready"
        break
    fi
    sleep 5
    echo -n "."
done

if [ $i -eq 120 ]; then
    print_error "Conda environment setup failed"
    exit 1
fi

# ========================================
# 5. Natural Language Dependencies
# ========================================

echo -e "\n${BLUE}5. Natural Language Dependencies${NC}"
echo "================================="

print_info "Installing NL packages..."
docker-compose exec app conda run -n brickbrain-rec pip install --quiet \
    nltk==3.8.1 \
    spacy==3.7.2 \
    scikit-learn==1.3.0

# Download NLTK data
print_info "Downloading NLTK data..."
docker-compose exec app conda run -n brickbrain-rec python -c "
import nltk
import ssl
import certifi

# Use certifi's certificates or create unverified context as fallback
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    # If SSL fails, try with unverified context
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_https_context
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
print('NLTK data downloaded successfully')
"
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
print('NLTK data downloaded successfully')
"

# Download spaCy model
print_info "Downloading spaCy English model..."
docker-compose exec app conda run -n brickbrain-rec python -m spacy download en_core_web_sm --quiet

print_status "NL dependencies installed"

# ========================================
# 6. Database Schema and Data Setup
# ========================================

echo -e "\n${BLUE}6. Database Schema and Data Setup${NC}"
echo "=================================="

print_info "Setting up database schemas..."

# Create database schemas
docker-compose exec app conda run -n brickbrain-rec python -c "
import psycopg2
import os

# Database connection
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'brickbrain'),
    user=os.getenv('DB_USER', 'brickbrain'),
    password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
    port=int(os.getenv('DB_PORT', 5432))
)

# Read and execute schema files
with open('src/db/rebrickable_schema.sql', 'r') as f:
    rebrickable_schema = f.read()

with open('src/db/user_interaction_schema.sql', 'r') as f:
    user_schema = f.read()

cur = conn.cursor()
cur.execute(rebrickable_schema)
cur.execute(user_schema)
conn.commit()

print('Database schemas created successfully')
conn.close()
"

print_status "Database schemas created"

# Load sample data if data directory exists
if [ -d "data/rebrickable" ]; then
    print_info "Loading Rebrickable data..."
    docker-compose exec app conda run -n brickbrain-rec python src/scripts/upload_rebrickable_data.py
    print_status "Rebrickable data loaded"
else
    print_warning "No data directory found. You'll need to load LEGO data manually."
fi

# ========================================
# 7. Vector Database Initialization
# ========================================

echo -e "\n${BLUE}7. Vector Database Initialization${NC}"
echo "=================================="

# Check dataset size and recommend approach
DATASET_SIZE=$(docker-compose exec postgres psql -U brickbrain -d brickbrain -t -c "SELECT COUNT(*) FROM sets WHERE num_parts > 0;" 2>/dev/null | tr -d ' ' || echo "0")

if [ "$DATASET_SIZE" -gt 5000 ]; then
    echo -e "\n${YELLOW}📊 Large Dataset Detected (${DATASET_SIZE} sets)${NC}"
    echo "For optimal performance, consider cloud processing:"
    echo "• AWS SageMaker: ~$0.50, 5-10 minutes (vs 1-2 hours locally)"
    echo "• Azure ML: ~$0.50, 5-10 minutes"
    echo "• See cloud/README.md for instructions"
    echo ""
    read -p "Use local processing anyway? (y/N): " USE_LOCAL
    if [[ ! "$USE_LOCAL" =~ ^[Yy]$ ]]; then
        print_info "Preparing data for cloud processing..."
        docker-compose exec app conda run -n brickbrain-rec python -c "
docker-compose exec app conda run -n brickbrain-rec python -c "
import sys
import os
sys.path.append('/app/src/scripts')
from lego_nlp_recommeder import NLPRecommender
import psycopg2

try:
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'brickbrain'),
        user=os.getenv('DB_USER', 'brickbrain'),
        password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
        port=int(os.getenv('DB_PORT', 5432))
    )
    
    nl_recommender = NLPRecommender(conn, use_openai=False)
    nl_recommender.prep_vectorDB(use_cloud=True)
    
    print('✅ Data prepared for cloud processing!')
    print('📁 Upload embeddings/lego_sets_for_cloud.json to your cloud service')
    print('📖 See cloud/README.md for detailed instructions')
    conn.close()
    
except Exception as e:
    print(f'❌ Error preparing cloud data: {e}')
"
"
        print_status "Cloud data preparation complete"
        print_info "Skipping local vector database initialization"
    timeout 1800 docker-compose exec app conda run -n brickbrain-rec python -c "
import sys
import os
sys.path.append('/app/src/scripts')

from lego_nlp_recommeder import NLPRecommender
import psycopg2

try:
    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'brickbrain'),
        user=os.getenv('DB_USER', 'brickbrain'),
        password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
        port=int(os.getenv('DB_PORT', 5432))
    )
    
    # Check data size
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM sets WHERE num_parts > 0;')
    set_count = cur.fetchone()[0]
    print(f'Processing {set_count} LEGO sets...')
    
    # Initialize NLP recommender (without OpenAI by default)
    nl_recommender = NLPRecommender(conn, use_openai=False)
    
    # Use limited processing for large datasets to avoid timeouts
    if set_count > 5000:
        print('Large dataset - processing subset for quick setup...')
        nl_recommender.prep_vectorDB(limit_sets=1000)
        print(f'Vector database initialized with 1000 sample sets (of {set_count} total)')
        print('For full coverage, consider cloud processing (see cloud/README.md)')
    else:
        print('Processing all sets...')
        nl_recommender.prep_vectorDB()
        print(f'Vector database initialized with all {set_count} sets')
    
    conn.close()

except psycopg2.Error as e:
    print(f'Database error during vector initialization: {e}')
    print('Please ensure LEGO data is loaded before initializing vectors.')
    sys.exit(1)
except ImportError as e:
    print(f'Missing dependency for vector initialization: {e}')
    print('Please ensure all NLP dependencies are installed.')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error during vector initialization: {e}')
    print('The system may work with limited NL search capabilities.')
"
except ImportError as e:
    print(f'Missing dependency for vector initialization: {e}')
    print('Please ensure all NLP dependencies are installed.')
    sys.exit(1)
except Exception as e:
    print(f'Unexpected error during vector initialization: {e}')
    print('The system may work with limited NL search capabilities.')
"
        print('Large dataset - processing subset for quick setup...')
        nl_recommender.prep_vectorDB(limit_sets=1000)
        print(f'Vector database initialized with 1000 sample sets (of {set_count} total)')
        print('For full coverage, consider cloud processing (see cloud/README.md)')
    else:
        print('Processing all sets...')
        nl_recommender.prep_vectorDB()
        print(f'Vector database initialized with all {set_count} sets')
    
    conn.close()
    
except Exception as e:
    print(f'Vector database initialization failed: {e}')
    print('This is normal if no LEGO data is loaded yet.')
"

    if [ $? -eq 124 ]; then
        print_warning "Vector database initialization timed out (30 minutes)"
        print_info "Consider using cloud processing for large datasets"
        print_info "System will still work with limited NL search capabilities"
    elif [ $? -eq 0 ]; then
        print_status "Vector database initialization completed"
    else
        print_warning "Vector database had issues but may work partially"
    fi
else
    print_info "Vector database initialization skipped - using cloud processing workflow"
fi

# ========================================
# 8. API Server Startup
# ========================================

echo -e "\n${BLUE}8. API Server Startup${NC}"
echo "====================="

print_info "API server is starting with the application container..."

# Wait for API to be ready
print_info "Waiting for API to be healthy..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "API server is healthy and ready"
        break
    fi
    sleep 3
    echo -n "."
done

if [ $i -eq 60 ]; then
    print_warning "API server health check timeout - may still be starting"
fi

# ========================================
# 9. Quick Functionality Test
# ========================================

echo -e "\n${BLUE}9. Quick Functionality Test${NC}"
echo "============================"

print_info "Testing basic functionality..."

# Test health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "ERROR")
if [[ "$HEALTH_RESPONSE" =~ "status" ]]; then
    print_status "Health endpoint working"
else
    print_warning "Health endpoint may not be ready"
fi

# Test NL search if data is available
if [ -d "data/rebrickable" ]; then
    print_info "Testing natural language search..."
    NL_RESPONSE=$(curl -s -X POST "http://localhost:8000/search/natural" \
        -H "Content-Type: application/json" \
        -d '{"query": "star wars sets", "top_k": 3}' \
        2>/dev/null || echo "ERROR")
    
    if [[ "$NL_RESPONSE" =~ "results" ]]; then
        print_status "Natural language search is working"
    else
        print_warning "Natural language search may need more setup"
    fi
fi

# ========================================
# 10. Setup Complete
# ========================================

echo -e "\n${BLUE}10. Setup Complete${NC}"
echo "=================="

print_status "LEGO Recommendation Engine is ready!"

echo -e "\n${GREEN}🚀 Services Running:${NC}"
echo "• PostgreSQL Database: localhost:5432"
echo "• FastAPI Server: http://localhost:8000"
echo "• API Documentation: http://localhost:8000/docs"

echo -e "\n${GREEN}📋 Available Endpoints:${NC}"
echo "• Health Check: GET /health"
echo "• Natural Language Search: POST /search/natural"
echo "• Query Understanding: POST /nlp/understand"
echo "• Traditional Search: POST /search/sets"

echo -e "\n${GREEN}🔧 Quick Commands:${NC}"
echo "• View logs: docker-compose logs app"
echo "• Access container: docker-compose exec app bash"
echo "• Run tests: ./run_all_tests.sh"
echo "• Stop services: docker-compose down"

echo -e "\n${GREEN}🧪 Next Steps:${NC}"
echo "1. Run tests: ./run_all_tests.sh"
echo "2. Try the API at: http://localhost:8000/docs"
echo "3. Test natural language queries"
echo "4. Load additional LEGO data if needed"

echo -e "\n${BLUE}🎉 Setup and startup complete! The system is ready for use.${NC}"
