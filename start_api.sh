#!/bin/bash

# Startup script for the LEGO Recommendation API

echo "🧱 Starting LEGO Recommendation API..."

# Set environment variables if not already set
export DB_HOST=${DB_HOST:-localhost}
export DB_PORT=${DB_PORT:-5432}
export DB_NAME=${DB_NAME:-brickbrain}
export DB_USER=${DB_USER:-brickbrain}
export DB_PASSWORD=${DB_PASSWORD:-brickbrain_password}

# Navigate to the scripts directory
cd "$(dirname "$0")/src/scripts"

# Start the FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
echo "Alternative docs at http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"

python -m uvicorn recommendation_api:app --host 0.0.0.0 --port 8000 --reload
