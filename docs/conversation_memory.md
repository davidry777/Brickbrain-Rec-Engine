# Conversation Memory Features for NLP Recommender

This document describes the conversation memory functionality added to the `NLPRecommender` class in the Brickbrain LEGO Recommendation Engine.

## Overview

The conversation memory system enables the NLP Recommender to maintain context across multiple user interactions, providing more personalized and contextually-aware recommendations.

## Key Features

### 1. Conversation Memory Storage
- **Memory Buffer**: Uses LangChain's `ConversationBufferMemory` to store chat history
- **User Context**: Maintains user preferences, search history, and feedback
- **Session Tracking**: Tracks queries and responses within a conversation session

### 2. Context-Aware Query Processing
- **Enhanced Intent Detection**: Uses conversation history to better understand user intent
- **Contextual Filter Extraction**: Applies learned preferences to filter extraction
- **Follow-up Query Handling**: Understands references to previous recommendations

### 3. User Preference Learning
- **Theme Preferences**: Learns from user feedback and search patterns
- **Piece Count Preferences**: Adapts to user's preferred set sizes
- **Feedback Integration**: Incorporates user likes/dislikes into future recommendations

## Core Classes and Methods

### ConversationContext
```python
@dataclass
class ConversationContext:
    user_preferences: Dict[str, Any]
    previous_recommendations: List[Dict[str, Any]]
    current_session_queries: List[str]
    follow_up_context: Dict[str, Any]
    conversation_summary: str
```

### Key Methods

#### Memory Management
- `add_to_conversation_memory(user_input, ai_response)`: Add interaction to memory
- `get_conversation_context()`: Get current conversation state
- `clear_conversation_memory()`: Reset memory and user context

#### Context-Aware Processing
- `process_nl_query_with_context(query)`: Process query with conversation context
- `semantic_search_with_context(query)`: Search with contextual awareness
- `get_conversational_recommendations(query)`: Full context-aware recommendations

#### User Preference Management
- `update_user_preferences(preferences)`: Update user preferences
- `record_user_feedback(set_num, feedback, rating)`: Record feedback on sets
- `handle_follow_up_query(query)`: Handle follow-up and reference queries

## Usage Examples

### Basic Conversation Flow
```python
# Initialize recommender
recommender = NLPRecommender(db_conn, use_openai=False)

# First query
query1 = "I'm looking for Star Wars sets for my 10-year-old nephew"
result1 = recommender.process_nl_query_with_context(query1)

# Add to memory
recommender.add_to_conversation_memory(query1, "Found Star Wars sets")

# Follow-up query (benefits from context)
query2 = "What about something similar but smaller?"
result2 = recommender.process_nl_query_with_context(query2)
```

### User Feedback Integration
```python
# Record user feedback
recommender.record_user_feedback('75309-1', 'liked', 5)
recommender.record_user_feedback('60321-1', 'not_interested', 2)

# Future queries will benefit from this feedback
query3 = "Show me another set"
result3 = recommender.process_nl_query_with_context(query3)
```

### Full Conversational Recommendations
```python
# Get recommendations with full context
recommendations = recommender.get_conversational_recommendations(
    "Find me a challenging build"
)

# Returns:
# {
#     'query': '...',
#     'results': [...],
#     'explanation': '...',
#     'conversation_context': {...},
#     'recommendation_metadata': {...}
# }
```

## User Context Structure

The user context maintains the following information:

```python
user_context = {
    'preferences': {
        'themes': {'Star Wars': 3, 'City': 1},  # Theme preferences with scores
        'piece_ranges': [500, 750, 1000],       # Preferred piece counts
        'recipient': 'nephew',                  # Gift recipient
        'age': 10                               # Age context
    },
    'previous_searches': [
        {
            'query': 'Star Wars sets for nephew',
            'timestamp': '2024-01-15T10:30:00',
            'response_summary': 'Found 5 Star Wars sets...'
        }
    ],
    'liked_sets': [
        {
            'set_num': '75309-1',
            'feedback': 'liked',
            'rating': 5,
            'timestamp': '2024-01-15T10:35:00'
        }
    ],
    'disliked_sets': [...],
    'conversation_session': '2024-01-15T10:00:00'
}
```

## Enhanced Query Processing

### Context-Aware Intent Detection
The system now considers conversation history when detecting intent:
- Follow-up queries are better understood
- References to previous recommendations are handled
- User's search patterns influence intent classification

### Contextual Filter Enhancement
- Applies learned preferences when filters aren't explicit
- Uses conversation history to fill in missing context
- Balances explicit filters with learned preferences

### Semantic Query Enhancement
The semantic query for embedding search is enhanced with:
- Previous conversation context
- User preferences
- Theme continuity from session
- Recipient and occasion context

## Benefits

1. **Improved Relevance**: Recommendations become more relevant as the system learns user preferences
2. **Better Follow-up Handling**: "Show me similar sets" queries work intuitively
3. **Personalized Experience**: Each user gets recommendations tailored to their history
4. **Context Continuity**: Conversations feel more natural and connected
5. **Learning from Feedback**: System improves over time based on user responses

## Demo Script

Run the conversation memory demo:
```bash
python examples/conversation_memory_demo.py
```

This demonstrates:
- Initial query processing with context extraction
- Follow-up query handling
- User feedback integration
- Preference learning
- Memory management

## Future Enhancements

Potential improvements to the conversation memory system:

1. **Conversation Summarization**: Use LLM to create more sophisticated conversation summaries
2. **Multi-Session Memory**: Persist user preferences across multiple sessions
3. **Collaborative Filtering**: Use preferences from similar users
4. **Emotion Detection**: Understand user sentiment and adjust recommendations
5. **Conversation Flow Management**: Better handling of complex conversation patterns

## Technical Notes

- Memory is stored in-memory by default (can be extended to persistent storage)
- Compatible with both OpenAI and local LLM setups
- Graceful fallback to regex-based processing when LLM is unavailable
- Conversation context is JSON-serializable for easy storage/retrieval
