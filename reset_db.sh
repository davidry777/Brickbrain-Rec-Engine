#!/bin/bash

# Stop containers
echo "🛑 Stopping Docker containers..."
docker-compose down

# Remove volume
echo "🗑️  Removing old database volume..."
docker volume rm brickbrain-rec-engine_postgres_data

# Start containers
echo "🚀 Starting Docker containers..."
docker-compose up -d

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Run the data upload script
echo "📥 Running data upload script..."
python src/scripts/upload_rebrickable_data.py

echo "✅ Process completed!"