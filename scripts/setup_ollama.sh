#!/bin/bash

# Setup script for Ollama inside Docker container
# This script can be called from the Docker container to set up Ollama

set -e

echo "🤖 Setting up Ollama for NLP functionality..."

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service in background
echo "🚀 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama service is ready"
        break
    fi
    sleep 2
done

if [ $i -eq 30 ]; then
    echo "❌ Ollama service failed to start"
    exit 1
fi

# Download Mistral model
echo "📦 Downloading Mistral model (this may take a few minutes)..."
if ollama pull mistral; then
    echo "✅ Mistral model downloaded successfully"
else
    echo "❌ Failed to download Mistral model"
    exit 1
fi

# Verify the model is available
echo "🔍 Verifying model availability..."
if ollama list | grep -q mistral; then
    echo "✅ Mistral model is ready for use"
else
    echo "❌ Mistral model verification failed"
    exit 1
fi

echo "🎉 Ollama setup complete!"
echo "📋 Available models:"
ollama list

# Keep Ollama running
echo "🔄 Keeping Ollama service running..."
wait $OLLAMA_PID
