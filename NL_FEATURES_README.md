# 🧱 Natural Language Features - Setup Guide

This guide helps you set up and test the natural language processing features of the LEGO Recommendation Engine.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the complete setup script
./setup_nl_features.sh

# Test the features
./test_nl_features.sh
```

### Option 2: Manual Setup
```bash
# 1. Start the infrastructure
docker-compose up -d

# 2. Wait for services to be ready
./reset_db.sh

# 3. Test the API
curl http://localhost:8000/health

# 4. Test natural language search
curl -X POST "http://localhost:8000/search/natural" \
     -H "Content-Type: application/json" \
     -d '{"query": "star wars sets for kids", "top_k": 5}'
```

## 🧠 Natural Language Features

### 1. Natural Language Search
Transform natural language queries into structured searches:

```python
# Example queries:
"star wars sets for kids under 10"
"birthday gift for my 8-year-old nephew"
"challenging technic sets between 1000-2000 pieces"
"detailed architecture sets under $200"
"something like the millennium falcon but smaller"
```

### 2. Query Understanding
Analyze and understand user intent:

```bash
curl -X POST "http://localhost:8000/nlp/understand" \
     -H "Content-Type: application/json" \
     -d '{"query": "star wars sets for kids"}'
```

### 3. Intent Detection
Automatically detects user intent:
- `product_search` - Looking for specific products
- `gift_recommendation` - Seeking gift suggestions
- `comparison` - Comparing different sets
- `budget_search` - Price-constrained searches
- `similar_recommendation` - Finding similar items

### 4. Filter Extraction
Automatically extracts search filters:
- **Themes**: "Star Wars", "City", "Technic", etc.
- **Age ranges**: "for kids", "8-year-old", "adults"
- **Piece counts**: "under 500 pieces", "1000-2000 pieces"
- **Price ranges**: "under $50", "between $100-200"
- **Complexity**: "simple", "challenging", "detailed"

### 5. Semantic Search
Uses vector embeddings for semantic similarity:
- Finds conceptually similar sets
- Understands context and relationships
- Provides relevance scoring

## 🔧 Configuration

### Environment Variables

The setup script creates these key environment variables:

```bash
# Natural Language Processing
TRANSFORMERS_CACHE=./.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=./.cache/sentence-transformers
LANGCHAIN_CACHE_DIR=./.cache/langchain
TOKENIZERS_PARALLELISM=false

# Optional: OpenAI Integration
OPENAI_API_KEY=your_api_key_here
```

### Dependencies

The NL features require these key packages:
- `langchain>=0.3.26` - LLM framework
- `sentence-transformers==5.0.0` - Embeddings
- `chromadb==1.0.15` - Vector database
- `transformers>=4.53.1` - Hugging Face models
- `nltk==3.8.1` - Natural language toolkit
- `spacy>=3.7.2` - Advanced NLP

## 📊 API Endpoints

### Natural Language Search
```http
POST /search/natural
Content-Type: application/json

{
  "query": "star wars sets for kids",
  "top_k": 5,
  "include_explanation": true,
  "user_id": 123
}
```

**Response:**
```json
{
  "query": "star wars sets for kids",
  "intent": "product_search",
  "extracted_filters": {
    "themes": ["Star Wars"],
    "max_age": 12,
    "complexity": "simple"
  },
  "results": [
    {
      "set_num": "75301",
      "name": "Luke Skywalker's X-wing Fighter",
      "theme": "Star Wars",
      "year": 2021,
      "num_parts": 474,
      "relevance_score": 0.95,
      "match_reasons": ["Star Wars theme", "Kid-friendly complexity"],
      "description": "Detailed X-wing fighter model..."
    }
  ],
  "explanation": "I found Star Wars sets that are perfect for kids...",
  "query_understanding": {
    "intent": "product_search",
    "confidence": 0.92,
    "entities": {
      "theme": "Star Wars",
      "recipient": "kids"
    }
  }
}
```

### Query Understanding
```http
POST /nlp/understand
Content-Type: application/json

{
  "query": "birthday gift for 8 year old who loves cars"
}
```

**Response:**
```json
{
  "original_query": "birthday gift for 8 year old who loves cars",
  "intent": "gift_recommendation",
  "confidence": 0.88,
  "extracted_filters": {
    "themes": ["City", "Speed Champions"],
    "max_age": 10,
    "min_age": 6
  },
  "extracted_entities": {
    "recipient": "8 year old",
    "occasion": "birthday",
    "interests": ["cars"]
  },
  "semantic_query": "birthday gift for 8 year old who loves cars from City, Speed Champions themes",
  "interpretation": "I understand you're looking for a birthday gift for an 8-year-old who loves cars..."
}
```

## 🧪 Testing

### Automated Tests
```bash
# Run all NL tests
./test_nl_features.sh

# Run specific test categories
docker-compose exec app conda run -n brickbrain-rec python tests/integration/nl_integration_test.py
docker-compose exec app conda run -n brickbrain-rec python tests/integration/final_validation.py
```

### Manual Testing
```bash
# Test with demo script
docker-compose exec app conda run -n brickbrain-rec python examples/nl_demo_script.py

# Test specific queries
curl -X POST "http://localhost:8000/search/natural" \
     -H "Content-Type: application/json" \
     -d '{"query": "detailed technic sets for adults", "top_k": 3}'
```

### Performance Testing
```bash
# Test response times
time curl -X POST "http://localhost:8000/search/natural" \
          -H "Content-Type: application/json" \
          -d '{"query": "star wars", "top_k": 10}'
```

## 🔍 Troubleshooting

### Common Issues

1. **"Natural language processor not initialized"**
   ```bash
   # Restart the application container
   docker-compose restart app
   
   # Check if vector database is initialized
   docker-compose exec app conda run -n brickbrain-rec python -c "
   import os
   print('Cache dir exists:', os.path.exists('.cache'))
   print('Embeddings dir exists:', os.path.exists('embeddings'))
   "
   ```

2. **No search results returned**
   ```bash
   # Check if LEGO data is loaded
   docker-compose exec app conda run -n brickbrain-rec python -c "
   import psycopg2
   conn = psycopg2.connect(host='postgres', database='brickbrain', user='brickbrain', password='brickbrain_password')
   cur = conn.cursor()
   cur.execute('SELECT COUNT(*) FROM sets')
   print('Sets in database:', cur.fetchone()[0])
   "
   ```

3. **Slow response times**
   ```bash
   # Check cache usage
   du -sh .cache/
   
   # Consider using GPU acceleration (if available)
   # Or reduce model size in lego_nlp_recommeder.py
   ```

### Debug Mode
```bash
# Enable debug logging
docker-compose exec app conda run -n brickbrain-rec python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Then run your tests
"
```

## 🚀 Advanced Configuration

### OpenAI Integration
For enhanced NL features, add your OpenAI API key:

```bash
# Add to .env file
OPENAI_API_KEY=your_openai_api_key_here
```

Then modify the NLP recommender initialization:
```python
# In recommendation_api.py
nl_recommender = NLPRecommender(get_db(), use_openai=True)
```

### Custom Models
To use different embedding models:

```python
# In lego_nlp_recommeder.py
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster
    # model_name="sentence-transformers/all-mpnet-base-v2",  # Better quality
)
```

### Performance Optimization
```python
# Add caching for frequent queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_semantic_search(query: str, top_k: int):
    return self.semantic_search(query, top_k)
```

## 📚 Development

### Adding New Features
1. **Custom Intent Detection**
   ```python
   # In lego_nlp_recommeder.py
   def _detect_intent(self, query: str) -> str:
       # Add your custom logic here
       if "compare" in query.lower():
           return "comparison"
       # ... existing logic
   ```

2. **New Filter Types**
   ```python
   # In SearchFilters class
   class SearchFilters(BaseModel):
       # Add new filter types
       color_preference: Optional[str] = None
       availability: Optional[str] = None
   ```

3. **Enhanced Explanations**
   ```python
   # Custom explanation generation
   def generate_custom_explanation(self, query: str, results: List[Dict]) -> str:
       # Your custom explanation logic
       pass
   ```

## 🎯 Best Practices

1. **Query Optimization**
   - Use specific terms: "Star Wars" instead of "space"
   - Include context: "for kids" or "for adults"
   - Be specific about requirements: "under $50" or "500+ pieces"

2. **Error Handling**
   - Always handle empty results gracefully
   - Provide fallback recommendations
   - Log failed queries for improvement

3. **Performance**
   - Cache frequent queries
   - Use appropriate top_k values
   - Monitor response times

4. **Testing**
   - Test with diverse queries
   - Include edge cases
   - Validate intent detection accuracy

## 📈 Monitoring

### Key Metrics
- Query response times
- Intent detection accuracy
- Result relevance scores
- Cache hit rates
- Error rates

### Logging
```python
# Add comprehensive logging
import logging

logger = logging.getLogger(__name__)

# Log query processing
logger.info(f"Processing query: {query}")
logger.debug(f"Extracted filters: {filters}")
logger.info(f"Found {len(results)} results")
```

## 🎉 Success!

Once setup is complete, you'll have:
- ✅ Natural language search capabilities
- ✅ Intelligent query understanding
- ✅ Automatic filter extraction
- ✅ Semantic similarity search
- ✅ Personalized recommendations
- ✅ Comprehensive API documentation

Visit [http://localhost:8000/docs](http://localhost:8000/docs) to explore the interactive API documentation and test the natural language features!
