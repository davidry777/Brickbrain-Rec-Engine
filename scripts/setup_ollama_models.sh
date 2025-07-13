#!/bin/bash

# Setup Ollama Models for LEGO Recommendation Engine
# This script downloads and prepares the required models for the NLP recommender

set -e  # Exit on any error

echo "🦙 Setting up Ollama models for LEGO Recommendation Engine..."

# Check if Ollama is running, if not start it
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "📡 Starting Ollama server..."
    
    # Start Ollama in background
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama is ready!"
            break
        fi
        echo "   Attempt $i/30: Waiting for Ollama..."
        sleep 2
    done
    
    # Check if Ollama is ready
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "❌ Failed to start Ollama after 60 seconds"
        exit 1
    fi
else
    echo "✅ Ollama is already running"
fi

# Download Mistral model
echo "📥 Downloading Mistral model (this may take a while for the first time)..."
if ollama pull mistral; then
    echo "✅ Mistral model downloaded successfully!"
else
    echo "❌ Failed to download Mistral model"
    exit 1
fi

# Verify the model is available
echo "🔍 Verifying Mistral model availability..."
if ollama list | grep -q mistral; then
    echo "✅ Mistral model is available and ready to use!"
else
    echo "❌ Mistral model verification failed"
    exit 1
fi

# Optional: Download other useful models
echo "🤔 Would you like to download additional models? (y/N)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "📥 Downloading additional models..."
    
    # Download a smaller, faster model for quick responses
    echo "📥 Downloading Llama 3.2 (3B) - smaller and faster model..."
    ollama pull llama3.2:3b
    
    # Download an embedding model (if needed)
    echo "📥 Downloading embedding model..."
    ollama pull nomic-embed-text
    
    echo "✅ Additional models downloaded!"
fi

echo "🎉 Ollama setup complete!"
echo ""
echo "📋 Available models:"
ollama list

echo ""
echo "🚀 You can now run the NLP recommender tests!"
echo "   Run: python -m tests.unit.test_nlp_recommender"
