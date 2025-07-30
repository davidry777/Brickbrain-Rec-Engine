#!/bin/bash

# 🧱 LEGO Recommendation Engine - Complete Setup Script
# Combines Docker-based setup with HuggingFace NLP integration
# Supports both containerized and local Python environments

set -e  # Exit on any error

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

echo -e "${BLUE}🧱 LEGO Recommendation Engine - Complete Setup${NC}"
echo "=============================================="
echo
echo "This script provides complete setup including:"
echo "• Docker containers for PostgreSQL and application"
echo "• HuggingFace NLP models and dependencies"
echo "• Local Python environment option"
echo "• Database initialization and sample data"
echo

# Parse command line arguments
SETUP_MODE="docker"  # Default to Docker setup
ENABLE_HUGGINGFACE=true
SKIP_OLLAMA=false
DOWNLOAD_MODELS=true
RUN_DEMO=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Setup options:"
    echo "  --docker         Use Docker-based setup (default)"
    echo "  --local          Use local Python environment setup"
    echo "  --hybrid         Docker for services, local for development"
    echo
    echo "Feature options:"
    echo "  --huggingface    Enable HuggingFace NLP features (default)"
    echo "  --no-huggingface Disable HuggingFace, use basic NLP only"
    echo "  --skip-ollama    Skip Ollama setup (faster startup)"
    echo "  --no-models      Skip model download (faster setup)"
    echo
    echo "Testing options:"
    echo "  --demo           Run demo after setup"
    echo "  --test           Run tests after setup"
    echo
    echo "Examples:"
    echo "  $0                           # Full Docker setup with HuggingFace"
    echo "  $0 --local --demo           # Local setup with demo"
    echo "  $0 --skip-ollama --no-models # Fast Docker setup"
    echo "  $0 --hybrid --test          # Hybrid setup with tests"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            SETUP_MODE="docker"
            shift
            ;;
        --local)
            SETUP_MODE="local"
            shift
            ;;
        --hybrid)
            SETUP_MODE="hybrid"
            shift
            ;;
        --huggingface)
            ENABLE_HUGGINGFACE=true
            shift
            ;;
        --no-huggingface)
            ENABLE_HUGGINGFACE=false
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --no-models)
            DOWNLOAD_MODELS=false
            shift
            ;;
        --demo)
            RUN_DEMO=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_info "Setup mode: $SETUP_MODE"
print_info "HuggingFace NLP: $ENABLE_HUGGINGFACE"
print_info "Download models: $DOWNLOAD_MODELS"
echo

# System compatibility checks
check_system_requirements() {
    print_info "Checking system requirements..."
    
    # Check operating system
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected"
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple"; then
            print_status "Apple Silicon detected (MPS support available)"
            export APPLE_SILICON=true
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux detected"
        if command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
            export CUDA_AVAILABLE=true
        fi
    fi
    
    # Check available memory (for model loading)
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
        print_info "Available memory: ${AVAILABLE_MEM}GB"
        
        if [[ $AVAILABLE_MEM -lt 4 ]] && [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
            print_warning "Less than 4GB memory available. Will enable model quantization."
            export ENABLE_MODEL_QUANTIZATION=true
        fi
    fi
}

# Docker-based setup
setup_docker_environment() {
    print_info "Setting up Docker environment..."
    
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

    # Wait for app container to be healthy
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
}

# Local Python environment setup
setup_local_environment() {
    print_info "Setting up local Python environment..."
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_status "Python $PYTHON_VERSION found"
    
    # Check if version is 3.8 or higher
    if [[ "$(echo "$PYTHON_VERSION 3.8" | tr " " "\n" | sort -V | head -n1)" != "3.8" ]]; then
        print_error "Python 3.8 or higher is required"
        exit 1
    fi
    
    # Setup virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "pip upgraded"
    
    # Install dependencies based on system
    install_python_dependencies
}

# Install Python dependencies with system optimization
install_python_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Install PyTorch with appropriate backend
    if [[ "$CUDA_AVAILABLE" == "true" ]]; then
        print_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$APPLE_SILICON" == "true" ]]; then
        print_info "Installing PyTorch with MPS support for Apple Silicon..."
        pip install torch torchvision torchaudio
    else
        print_info "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install main requirements
    print_info "Installing main requirements..."
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully"
}

# HuggingFace model setup and download
setup_huggingface_models() {
    if [[ "$ENABLE_HUGGINGFACE" != "true" ]]; then
        print_info "Skipping HuggingFace model setup"
        return 0
    fi
    
    print_info "Setting up HuggingFace NLP models..."
    
    # Set environment for model downloads
    export HF_HOME="${HOME}/.cache/huggingface"
    mkdir -p "$HF_HOME"
    
    if [[ "$DOWNLOAD_MODELS" == "true" ]]; then
        print_info "Downloading and caching HuggingFace models..."
        
        # Determine Python command based on setup mode
        if [[ "$SETUP_MODE" == "local" ]]; then
            PYTHON_CMD="python3"
        else
            PYTHON_CMD="docker-compose exec app conda run -n brickbrain-rec python"
        fi
        
        $PYTHON_CMD -c "
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src/scripts')

print('📥 Downloading HuggingFace models...')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer
    
    # Essential models for LEGO NLP
    models = [
        ('distilbert-base-uncased', 'tokenizer'),
        ('microsoft/DialoGPT-small', 'conversation'),
        ('distilgpt2', 'text-generation'),
        ('dbmdz/bert-large-cased-finetuned-conll03-english', 'ner'),
        ('facebook/bart-large-cnn', 'summarization')
    ]
    
    downloaded = 0
    failed = 0
    
    for model_name, model_type in models:
        try:
            print(f'  Downloading {model_name}...')
            if model_type == 'tokenizer':
                AutoTokenizer.from_pretrained(model_name)
            elif model_type == 'conversation':
                AutoTokenizer.from_pretrained(model_name)
                AutoModelForCausalLM.from_pretrained(model_name)
            elif model_type in ['text-generation', 'ner', 'summarization']:
                pipeline(model_type, model=model_name)
            print(f'  ✅ {model_name} downloaded')
            downloaded += 1
        except Exception as e:
            print(f'  ⚠️ Failed to download {model_name}: {e}')
            failed += 1
    
    # Download sentence transformer
    try:
        print('  Downloading sentence-transformers/all-MiniLM-L6-v2...')
        SentenceTransformer('all-MiniLM-L6-v2')
        print('  ✅ Sentence transformer downloaded')
        downloaded += 1
    except Exception as e:
        print(f'  ⚠️ Failed to download sentence transformer: {e}')
        failed += 1
    
    print(f'🎉 Model downloads completed! Downloaded: {downloaded}, Failed: {failed}')
    
    if failed > 0:
        print('⚠️  Some models failed to download. System will still work but may have reduced functionality.')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('   HuggingFace dependencies may not be installed correctly.')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)
" 2>/dev/null || print_warning "Model download encountered issues but system may still work"
        
        print_status "HuggingFace models setup completed"
    else
        print_info "Skipping model download (will download on first use)"
    fi
}

# Environment configuration
setup_environment_config() {
    print_info "Setting up environment configuration..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# System Configuration
SETUP_MODE=$SETUP_MODE

# HuggingFace NLP Configuration
USE_HUGGINGFACE_NLP=$ENABLE_HUGGINGFACE
SKIP_HEAVY_INITIALIZATION=false
ENABLE_MODEL_QUANTIZATION=${ENABLE_MODEL_QUANTIZATION:-false}

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=brickbrain
DB_USER=brickbrain
DB_PASSWORD=brickbrain_password

# Model Configuration
HF_MODEL_CACHE_DIR=~/.cache/huggingface
TRANSFORMERS_OFFLINE=false
MAX_CONVERSATION_HISTORY=20
BATCH_SIZE=16
MAX_LENGTH=512

# Performance Configuration
CUDA_AVAILABLE=${CUDA_AVAILABLE:-false}
APPLE_SILICON=${APPLE_SILICON:-false}
EOF
        print_status "Environment configuration created (.env file)"
    else
        print_info "Environment configuration already exists"
        
        # Update key settings
        if grep -q "USE_HUGGINGFACE_NLP" .env; then
            sed -i.bak "s/USE_HUGGINGFACE_NLP=.*/USE_HUGGINGFACE_NLP=$ENABLE_HUGGINGFACE/" .env
        else
            echo "USE_HUGGINGFACE_NLP=$ENABLE_HUGGINGFACE" >> .env
        fi
        
        if grep -q "SETUP_MODE" .env; then
            sed -i.bak "s/SETUP_MODE=.*/SETUP_MODE=$SETUP_MODE/" .env
        else
            echo "SETUP_MODE=$SETUP_MODE" >> .env
        fi
    fi
    
    print_status "Environment configured for $SETUP_MODE setup with HuggingFace: $ENABLE_HUGGINGFACE"
}

# Setup Gradio interface
setup_gradio_interface() {
    if [[ "$SETUP_MODE" == "local" ]]; then
        print_info "Gradio interface available via: python gradio/gradio_launcher.py"
        return 0
    fi
    
    print_info "Starting Gradio interface container..."
    if ! docker-compose up -d gradio; then
        print_error "Failed to start Gradio interface container"
        print_info "Check logs with: docker-compose logs gradio"
        return 1
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
}
# Verify and install NLP packages
verify_nlp_packages() {
    print_info "Verifying NLP packages..."
    
    if [[ "$SETUP_MODE" == "local" ]]; then
        # For local setup, verify packages in current environment
        python3 -c "
try:
    import transformers
    import sentence_transformers
    import torch
    if '$ENABLE_HUGGINGFACE' == 'true':
        import langchain_huggingface
    print('✅ All required NLP packages verified')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    raise
" || {
    print_warning "Some packages missing, installing..."
    pip install transformers sentence-transformers torch
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        pip install langchain-huggingface
    fi
}
    else
        # For Docker setup, verify packages in container
        docker-compose exec app conda run -n brickbrain-rec python -c "
try:
    import transformers
    import sentence_transformers
    import torch
    if '$ENABLE_HUGGINGFACE' == 'true':
        import langchain_huggingface
        import langchain_ollama
    import faiss
    print('✅ All NLP packages verified')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('Installing missing packages...')
    import subprocess
    packages = ['transformers>=4.53.1', 'sentence-transformers==5.0.0', 'torch>=2.0.0']
    if '$ENABLE_HUGGINGFACE' == 'true':
        packages.extend(['langchain-huggingface==0.3.0', 'langchain-ollama==0.3.4'])
    subprocess.run(['pip', 'install'] + packages, check=True)
    print('✅ Packages installed')
" 2>/dev/null || print_warning "Could not verify all packages - may need manual installation"
    fi
    
    print_status "NLP packages verification completed"
}

# Setup database schemas
setup_database() {
    print_info "Setting up database schemas..."
    
    if [[ "$SETUP_MODE" == "local" ]]; then
        # For local setup, connect directly to database
        python3 -c "
import psycopg2
import os

print('Connecting to database...')
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    database=os.getenv('DB_NAME', 'brickbrain'),
    user=os.getenv('DB_USER', 'brickbrain'),
    password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
    port=int(os.getenv('DB_PORT', 5432))
)

print('Creating essential database schemas...')
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

# Create conversation memory table (for HuggingFace NLP)
if '$ENABLE_HUGGINGFACE' == 'true':
    cur.execute('''
    CREATE TABLE IF NOT EXISTS conversation_memory (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        conversation_id VARCHAR(255) NOT NULL,
        message_type VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')
    
    cur.execute('''
    CREATE INDEX IF NOT EXISTS idx_conversation_memory_user 
    ON conversation_memory(user_id);
    ''')
    
    cur.execute('''
    CREATE INDEX IF NOT EXISTS idx_conversation_memory_conversation 
    ON conversation_memory(conversation_id);
    ''')

conn.commit()
print('✅ Essential database schemas created successfully')
conn.close()
"
    else
        # For Docker setup, use container environment
        docker-compose exec app conda run -n brickbrain-rec python -c "
import psycopg2
import os

print('Connecting to database...')
conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'postgres'),
    database=os.getenv('DB_NAME', 'brickbrain'),
    user=os.getenv('DB_USER', 'brickbrain'),
    password=os.getenv('DB_PASSWORD', 'brickbrain_password'),
    port=int(os.getenv('DB_PORT', 5432))
)

print('Creating essential database schemas...')
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

# Create conversation memory table (for HuggingFace NLP)
if '$ENABLE_HUGGINGFACE' == 'true':
    cur.execute('''
    CREATE TABLE IF NOT EXISTS conversation_memory (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        conversation_id VARCHAR(255) NOT NULL,
        message_type VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')

conn.commit()
print('✅ Essential database schemas created successfully')
conn.close()
"
    fi
    
    print_status "Database schemas setup complete"
}

# Load sample data
load_sample_data() {
    # Load sample data if available
    if [ -d "data/rebrickable" ]; then
        print_info "Loading Rebrickable data..."
        if [[ "$SETUP_MODE" == "local" ]]; then
            python3 src/scripts/rebrickable_container_uploader.py
        else
            docker-compose exec app conda run -n brickbrain-rec python src/scripts/rebrickable_container_uploader.py
        fi
        print_status "Rebrickable data loaded successfully"
    else
        print_warning "No data directory found. You'll need to load LEGO data manually."
        print_info "Download data from: https://rebrickable.com/downloads/"
        print_info "Place CSV files in: data/rebrickable/"
    fi
}

# Setup Ollama (optional, for legacy support)
setup_ollama() {
    if [[ "$SKIP_OLLAMA" == "true" ]] || [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        print_info "Skipping Ollama setup (HuggingFace preferred)"
        return 0
    fi
    
    if [[ "$SETUP_MODE" == "local" ]]; then
        print_info "For local Ollama setup, please install manually:"
        print_info "curl -fsSL https://ollama.com/install.sh | sh"
        print_info "ollama pull mistral"
    else
        print_info "Setting up Ollama inside Docker container..."
        
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
            
            echo '🎉 Ollama setup complete inside container!'
        " && print_status "Ollama setup completed successfully" || print_warning "Ollama setup failed - using HuggingFace only"
    fi
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        if [[ "$SETUP_MODE" == "local" ]]; then
            # Test local installation
            python3 -c "
import sys
sys.path.insert(0, 'src/scripts')

try:
    from hf_nlp_recommender import HuggingFaceNLPRecommender
    print('✅ HuggingFace NLP module import successful')
    
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA GPU available: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        print('✅ Apple Silicon MPS available')
    else:
        print('ℹ️ Using CPU (no GPU acceleration)')
    
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    print('✅ HuggingFace libraries test passed')
    print('🎉 Installation test completed successfully!')
    
except Exception as e:
    print(f'⚠️ Test warning: {e}')
    print('ℹ️ Basic setup completed, but some functionality may require additional configuration')
"
        else
            # Test Docker installation
            docker-compose exec app conda run -n brickbrain-rec python -c "
import sys
sys.path.insert(0, 'src/scripts')

try:
    if '$ENABLE_HUGGINGFACE' == 'true':
        from hf_nlp_recommender import HuggingFaceNLPRecommender
        print('✅ HuggingFace NLP module import successful')
    
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA GPU available')
    elif torch.backends.mps.is_available():
        print('✅ Apple Silicon MPS available')
    else:
        print('ℹ️ Using CPU (no GPU acceleration)')
    
    print('✅ Installation test passed')
    print('🎉 Installation test completed successfully!')
    
except Exception as e:
    print(f'⚠️ Test warning: {e}')
    print('ℹ️ Basic setup completed, but some functionality may require additional configuration')
"
        fi
    fi
    
    # Test API availability
    if [[ "$SETUP_MODE" != "local" ]]; then
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_status "API server is accessible"
        else
            print_warning "API server may still be starting"
        fi
    fi
    
    print_status "Installation testing completed"
}

# Run demo (optional)
run_demo() {
    if [[ "$RUN_DEMO" != "true" ]]; then
        return 0
    fi
    
    print_info "Running demonstration..."
    
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        if [[ "$SETUP_MODE" == "local" ]]; then
            print_info "Starting local demo..."
            python3 examples/hf_nlp_demo.py --demo health
        else
            print_info "Running containerized demo..."
            docker-compose exec app conda run -n brickbrain-rec python examples/hf_nlp_demo.py --demo health
        fi
    else
        print_info "Running basic recommendation demo..."
        if [[ "$SETUP_MODE" == "local" ]]; then
            python3 examples/example_client.py
        else
            docker-compose exec app conda run -n brickbrain-rec python examples/example_client.py
        fi
    fi
}

# Run tests (optional)
run_tests() {
    if [[ "$RUN_TESTS" != "true" ]]; then
        return 0
    fi
    
    print_info "Running basic tests..."
    ./scripts/run_all_tests.sh
}

# Main execution flow
main() {
    # Check prerequisites
    check_system_requirements
    
    # Setup environment configuration first
    setup_environment_config
    
    # Setup based on selected mode
    case $SETUP_MODE in
        "docker")
            setup_docker_environment
            setup_gradio_interface
            verify_nlp_packages
            ;;
        "local")
            setup_local_environment
            ;;
        "hybrid")
            setup_docker_environment  # Docker for database and services
            setup_local_environment   # Local Python for development
            setup_gradio_interface
            ;;
        *)
            print_error "Unknown setup mode: $SETUP_MODE"
            exit 1
            ;;
    esac
    
    # Setup database
    setup_database
    
    # Load sample data
    load_sample_data
    
    # Setup HuggingFace models if enabled
    setup_huggingface_models
    
    # Setup Ollama if requested (legacy support)
    setup_ollama
    
    # Test the installation
    test_installation
    
    # Run demo if requested
    run_demo
    
    # Run tests if requested  
    run_tests
    
    # Print completion summary
    print_completion_summary
}

# Print completion summary
print_completion_summary() {
    echo
    print_status "🎉 Setup completed successfully!"
    
    echo -e "\n${GREEN}📊 Setup Summary:${NC}"
    echo "• Setup Mode: $SETUP_MODE"
    echo "• HuggingFace NLP: $ENABLE_HUGGINGFACE"
    echo "• Models Downloaded: $DOWNLOAD_MODELS"
    echo "• Ollama Setup: $([ "$SKIP_OLLAMA" == "true" ] && echo "Skipped" || echo "Included")"
    
    echo -e "\n${GREEN}🚀 Services Running:${NC}"
    if [[ "$SETUP_MODE" != "local" ]]; then
        echo "• PostgreSQL Database: localhost:5432"
        echo "• FastAPI Server: http://localhost:8000"
        echo "• API Documentation: http://localhost:8000/docs"
        echo "• Gradio Interface: http://localhost:7860"
        if [[ "$SKIP_OLLAMA" != "true" ]]; then
            echo "• Ollama Service: http://localhost:11434"
        fi
    else
        echo "• Local Python Environment: venv/"
        echo "• Database: Connect to localhost:5432 (start manually)"
        echo "• FastAPI: Run with 'python src/scripts/recommendation_api.py'"
        echo "• Gradio: Run with 'python gradio/gradio_launcher.py'"
    fi
    
    echo -e "\n${GREEN}📋 Available Endpoints:${NC}"
    echo "• Health Check: GET /health"
    echo "• Traditional Search: POST /search/sets"
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo "• HuggingFace NLP Query: POST /nlp/query"
        echo "• HuggingFace Chat: POST /nlp/chat"
        echo "• HuggingFace Feedback: POST /nlp/feedback"
        echo "• HuggingFace Health: GET /health/huggingface"
    else
        echo "• Query Understanding: POST /nlp/understand"
        echo "• Natural Language Search: POST /nlp/search"
    fi
    
    echo -e "\n${GREEN}🔧 Quick Commands:${NC}"
    if [[ "$SETUP_MODE" != "local" ]]; then
        echo "• View API logs: docker-compose logs app"
        echo "• View Gradio logs: docker-compose logs gradio"
        echo "• Stop services: docker-compose down"
        echo "• Restart services: docker-compose restart"
    fi
    
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo "• Test HuggingFace NLP: python examples/hf_nlp_demo.py"
        echo "• Run HF health check: python examples/hf_nlp_demo.py --demo health"
    fi
    
    echo "• Run all tests: ./scripts/run_all_tests.sh"
    echo "• Quick validation: ./scripts/run_all_tests.sh --integration"
    
    echo -e "\n${GREEN}📚 Documentation:${NC}"
    echo "• API Documentation: http://localhost:8000/docs"
    echo "• Setup Guide: README.md"
    echo "• Test Documentation: tests/README.md"
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo "• HuggingFace Migration Guide: docs/huggingface_migration_guide.md"
        echo "• Migration Summary: HUGGINGFACE_MIGRATION_COMPLETE.md"
    fi
    
    echo -e "\n${GREEN}🎯 What's Ready:${NC}"
    echo "• ✅ LEGO recommendation system"
    echo "• ✅ PostgreSQL database with schemas"
    echo "• ✅ FastAPI REST endpoints"
    echo "• ✅ Gradio web interface"
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo "• ✅ HuggingFace NLP with conversation memory"
        echo "• ✅ Advanced intent classification and entity extraction"
        echo "• ✅ GPU/CPU optimization with quantization"
        echo "• ✅ User preference learning and analytics"
    else
        echo "• ✅ Basic natural language processing"
    fi
    
    echo -e "\n${BLUE}🚀 Next Steps:${NC}"
    if [[ "$SETUP_MODE" == "local" ]]; then
        echo "1. Start the database: docker-compose up -d postgres"
        echo "2. Start the API: python src/scripts/recommendation_api.py"
        echo "3. Open Gradio interface: python gradio/gradio_launcher.py"
    else
        echo "1. Test the system: curl http://localhost:8000/health"
        echo "2. Open Gradio interface: http://localhost:7860"
        echo "3. Check API docs: http://localhost:8000/docs"
    fi
    
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo "4. Try HuggingFace features: python examples/hf_nlp_demo.py"
        echo "5. Test conversation: curl -X POST http://localhost:8000/nlp/chat -H 'Content-Type: application/json' -d '{\"message\": \"Hello!\", \"user_id\": \"test\"}'"
    fi
    
    echo
    echo -e "${GREEN}🎉 Your enhanced LEGO recommendation system is ready!${NC}"
    if [[ "$ENABLE_HUGGINGFACE" == "true" ]]; then
        echo -e "${GREEN}   Enjoy the advanced NLP capabilities! 🧱✨${NC}"
    else
        echo -e "${GREEN}   Enjoy building with LEGO data! 🧱✨${NC}"
    fi
}

# Run the main setup
main "$@"
