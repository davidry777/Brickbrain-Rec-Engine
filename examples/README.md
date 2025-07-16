# Examples

This directory contains example scripts demonstrating various features of the LEGO NLP Recommender system.

## Available Examples

### 1. Basic Client Example (`example_client.py`)
A basic client demonstrating how to interact with the API endpoints.

### 2. Natural Language Demo (`nl_demo_script.py`)
Shows how to use the natural language processing features for LEGO set recommendations.

### 3. Conversation Memory Demo (`conversation_memory_demo.py`)
A comprehensive demonstration of the conversation memory capabilities including:
- Context-aware conversation flow
- User preference learning from feedback
- Follow-up query understanding
- Conversational AI with memory
- Memory management and cleanup

## Running the Examples

1. Ensure the API server is running:
   ```bash
   python src/scripts/recommendation_api.py
   ```

2. Run any example script:
   ```bash
   python examples/example_client.py
   python examples/nl_demo_script.py
   python examples/conversation_memory_demo.py
   ```

## Conversation Memory Demo Features

The conversation memory demo showcases:

### Context-Aware Conversations
- Maintains conversation history across multiple queries
- Understands follow-up questions and references
- Provides contextual recommendations based on previous interactions

### User Preference Learning
- Records user feedback on recommendations
- Learns from user ratings and preferences
- Adapts future recommendations based on learned preferences

### Follow-Up Query Processing
- Handles ambiguous queries using conversation context
- Processes relative references ("show me similar sets", "something smaller")
- Maintains conversation flow naturally

### Memory Management
- Tracks conversation state and history
- Provides memory inspection and debugging
- Supports memory clearing and reset

## API Endpoints Used

The examples demonstrate usage of various API endpoints:
- `/search/natural` - Natural language search
- `/recommendations/conversational` - Conversational recommendations
- `/ai/chat` - Conversational AI with memory
- `/ai/follow-up` - Follow-up query processing
- `/nlp/conversation/memory` - Memory management
- `/nlp/conversation/feedback` - User feedback recording

## Prerequisites

- Python 3.8+
- Required packages: `requests`, `json`
- Running API server
- Properly configured database with LEGO data
