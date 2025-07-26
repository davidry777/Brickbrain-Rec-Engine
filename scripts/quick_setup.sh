#!/bin/bash

# Temporary fix for testing the NLP recommender
# Run this instead of the full setup script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo -e "${BLUE}🧱 LEGO Recommendation Engine - Quick Setup for Testing${NC}"
echo "======================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# Clean setup
print_info "Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Create required volumes
print_info "Creating required Docker volumes..."
docker volume create conda_envs 2>/dev/null || print_info "Volume conda_envs already exists"

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

print_info "Starting application container..."
docker-compose up -d app

# Wait for app container to be healthy (uses Docker's built-in health check)
print_info "Waiting for app container to be healthy..."
for i in {1..60}; do
    health_status=$(docker inspect --format='{{.State.Health.Status}}' brickbrain-app 2>/dev/null || echo "starting")
    if [ "$health_status" = "healthy" ]; then
        print_status "App container is healthy and FastAPI server is running"
        break
    elif [ "$health_status" = "unhealthy" ]; then
        print_error "App container health check failed"
        print_info "Check logs with: docker-compose logs app"
        exit 1
    fi
    sleep 5
    echo -n "."
done

if [ $i -eq 60 ]; then
    print_error "App container failed to become healthy within 5 minutes"
    print_info "Check logs with: docker-compose logs app"
    exit 1
fi

# Start Gradio interface container
print_info "Starting Gradio interface container..."
if ! docker-compose up -d gradio; then
    print_error "Failed to start Gradio interface container"
    print_info "Check logs with: docker-compose logs gradio"
    exit 1
fi

# Wait for Gradio container to be healthy
print_info "Waiting for Gradio interface to be ready..."
for i in {1..30}; do
    gradio_health=$(docker inspect --format='{{.State.Health.Status}}' brickbrain-gradio 2>/dev/null || echo "starting")
    if [ "$gradio_health" = "healthy" ]; then
        print_status "Gradio interface is ready at http://localhost:7860"
        break
    elif [ "$gradio_health" = "unhealthy" ]; then
        print_warning "Gradio container health check failed, but it may still be starting up"
        break
    fi
    sleep 10
    echo -n "."
done

if [ $i -eq 30 ]; then
    print_warning "Gradio container didn't become healthy within 5 minutes, but may still be starting"
    print_info "Check status with: docker-compose logs gradio"
fi

# Verify NLP packages are installed in container
print_info "Verifying NLP packages in container..."
docker-compose exec app conda run -n brickbrain-rec python -c "
try:
    import langchain_huggingface
    import langchain_ollama
    import sentence_transformers
    import faiss
    print('✅ All NLP packages verified')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('Installing missing packages...')
    import subprocess
    subprocess.run(['pip', 'install', 'langchain-huggingface==0.3.0', 'langchain-ollama==0.3.4'], check=True)
    print('✅ Packages installed')
" 2>/dev/null || print_warning "Could not verify all packages - may need manual installation"

# Setup database schemas (optimized)
print_info "Setting up database schemas..."
docker-compose exec app conda run -n brickbrain-rec python -c "
import psycopg2
import os

print('Connecting to database...')
# Database connection
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'brickbrain'),
    user=os.getenv('DB_USER', 'brickbrain'),
    password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
    port=int(os.getenv('DB_PORT', 5432))
)

print('Creating minimal user interaction schema...')
# Create only essential tables to avoid hanging on large schema
cur = conn.cursor()

# Create basic users table
cur.execute('''
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

# Create basic user interactions table
cur.execute('''
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    set_num VARCHAR(20),
    interaction_type VARCHAR(20),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
''')

conn.commit()
print('✅ Essential user interaction schema created successfully')
print('📋 Additional tables will be created as needed')
conn.close()
"

print_status "Database schemas setup complete"

# Load sample data if available
if [ -d "data/rebrickable" ]; then
    print_info "Loading Rebrickable data with optimized uploader..."
    docker-compose exec app conda run -n brickbrain-rec python src/scripts/rebrickable_container_uploader.py
    print_status "Rebrickable data loaded successfully"
else
    print_warning "No data directory found. You'll need to load LEGO data manually."
fi

# Setup Ollama inside Docker container (optional for faster startup)
if [ "${SKIP_OLLAMA:-false}" != "true" ]; then
    print_info "Setting up Ollama inside Docker container..."

    # Run the Ollama setup script inside the container
    print_info "Installing Ollama in container..."
    docker-compose exec app bash -c "
        echo '📥 Installing curl first...'
        apt-get update && apt-get install -y curl
        
        echo '🤖 Installing Ollama...'
        curl -fsSL https://ollama.com/install.sh | sh
        
        echo '🚀 Starting Ollama service...'
        ollama serve &
        OLLAMA_PID=\$!
        
        echo '⏳ Waiting for Ollama to be ready...'
        for i in {1..30}; do
            if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo '✅ Ollama service is ready'
                break
            fi
            sleep 2
        done
        
        if [ \$i -eq 30 ]; then
            echo '❌ Ollama service failed to start'
            exit 1
        fi
        
        echo '📦 Downloading Mistral model...'
        if ollama pull mistral; then
            echo '✅ Mistral model downloaded successfully'
        else
            echo '❌ Failed to download Mistral model'
            exit 1
        fi
        
        echo '🔍 Verifying model availability...'
        if ollama list | grep -q mistral; then
            echo '✅ Mistral model is available and ready to use'
        else
            echo '❌ Mistral model verification failed'
            exit 1
        fi
        
        echo '🎉 Ollama setup complete inside container!'
        ollama list
    "

    if [ $? -eq 0 ]; then
        print_status "Ollama setup completed successfully in container"
    else
        print_warning "Ollama setup failed - NLP features may not work"
    fi
else
    print_info "Skipping Ollama setup (SKIP_OLLAMA=true). Basic NLP features will work without LLM."
fi

print_status "Quick setup complete!"
print_info "You can now test your NLP recommender with:"
print_info "python -m tests.unit.test_nlp_recommender"

echo -e "\n${GREEN}🚀 Services Running:${NC}"
echo "• PostgreSQL Database: localhost:5432"
echo "• FastAPI Server: http://localhost:8000"
echo "• API Documentation: http://localhost:8000/docs"
echo "• Gradio Interface: http://localhost:7860"
echo "• Ollama Service: http://localhost:11434"
echo "• Application Container: ready"

echo -e "\n${GREEN}📋 Available Endpoints:${NC}"
echo "• Health Check: GET /health"
echo "• Traditional Search: POST /search/sets"
echo "• Query Understanding: POST /nlp/understand"
echo "• Natural Language Search: POST /nlp/search"

echo -e "\n${GREEN}🔧 Quick Commands:${NC}"
echo "• Test NLP: python -m tests.unit.test_nlp_recommender"
echo "• Test Gradio Setup: python -m tests.unit.test_gradio_setup"
echo "• Validate Setup: python -m tests.integration.validate_nlp_setup"
echo "• API Docs: http://localhost:8000/docs"
echo "• Gradio Interface: http://localhost:7860"
echo "• View logs: docker-compose logs app"
echo "• View Gradio logs: docker-compose logs gradio"
echo "• Stop services: docker-compose down"
echo "• Check Ollama: ollama list"

echo -e "\n${GREEN}🤖 NLP Features Ready:${NC}"
echo "• Natural language query processing"
echo "• Intent detection and entity extraction"
echo "• Semantic search with embeddings"
echo "• Mistral LLM for query understanding"

echo -e "\n${YELLOW}⚠️  Note:${NC}"
echo "• All features including NLP search are now ready to use"
echo "• Vector database will be initialized on first NLP query"
echo "• First query may take longer due to model initialization"
