#!/bin/bash

# Brickbrain Gradio Interface Launch Script
# This script helps you launch the Gradio interface for the LEGO recommendation system

set -e

echo "🧱 Brickbrain Gradio Interface Setup"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Docker is running
print_status "Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_success "Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose not found. Please install docker-compose."
    exit 1
fi

# Check if the API services are running
print_status "Checking if Brickbrain services are running..."
if docker-compose ps | grep -q "brickbrain-app.*Up"; then
    print_success "Brickbrain services are running"
else
    print_warning "Brickbrain services not running. Starting them now..."
    
    print_status "Starting Docker Compose services..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Wait for API to be healthy
    print_status "Waiting for API to be healthy..."
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_success "API is healthy!"
            break
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "API failed to become healthy. Check logs with: docker-compose logs app"
        exit 1
    fi
fi

# Check if gradio is installed in the environment
print_status "Checking Gradio installation..."
if docker-compose exec -T app conda run -n brickbrain-rec python -c "import gradio" 2>/dev/null; then
    print_success "Gradio is already installed"
else
    print_warning "Installing Gradio in the container..."
    docker-compose exec app conda run -n brickbrain-rec pip install gradio>=4.0.0
    print_success "Gradio installed successfully"
fi

# Launch the Gradio interface
print_status "Launching Gradio interface..."
echo ""
echo "🌐 Gradio Interface will be available at:"
echo "   http://localhost:7860"
echo ""
echo "📡 API Documentation is available at:"
echo "   http://localhost:8000/docs"
echo ""
echo "🔍 To stop the interface, press Ctrl+C"
echo "🔧 To stop all services: docker-compose down"
echo ""

# Run the gradio interface in the container
docker-compose exec app conda run -n brickbrain-rec python /app/gradio/gradio_interface.py
