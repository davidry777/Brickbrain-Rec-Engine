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
sleep 15

# Run the data upload script inside the app container with correct DB_HOST
echo "📥 Running data upload script..."
if docker exec -e DB_HOST=postgres brickbrain-app conda run -n brickbrain-rec python src/scripts/upload_rebrickable_data.py; then
    echo "✅ Data upload completed successfully!"
    
    # Set up user interaction schema (only after main schema and data are loaded)
    echo "👥 Setting up user interaction schema..."
    docker exec -i brickbrain-postgres psql -U brickbrain -d brickbrain < src/db/user_interaction_schema.sql
    
    echo "✅ Process completed!"
else
    echo "❌ Data upload failed! Skipping user interaction schema setup."
    exit 1
fi