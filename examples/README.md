# Examples for LEGO Recommendation System

This directory contains example code and demonstrations of how to use the LEGO Recommendation System.

## Available Examples

### 📱 API Client Example (`example_client.py`)
**Purpose**: Demonstrates complete API workflow from user creation to getting recommendations

**Features**:
- User registration and management
- Set rating and interaction tracking
- Personalized recommendations
- Search functionality
- Wishlist management
- API metrics retrieval

**Usage**:
```bash
# Start the API first
python src/scripts/recommendation_api.py

# Run the example client
python examples/example_client.py
```

**What it demonstrates**:
1. ✅ Health check and API connectivity
2. 👥 User creation and profile management
3. ⭐ Rating sets and recording interactions
4. 🎯 Getting personalized recommendations
5. 🔍 Searching for specific sets
6. 💝 Managing wishlists and collections
7. 📊 Retrieving API metrics

## Expected Output

When you run `example_client.py`, you should see:

```
🧱 LEGO Recommendation API Demo
========================================

1. Checking API health...
✅ API Status: healthy

2. Getting available themes...
📂 Found 479 themes

3. Searching for Star Wars sets...
🔍 Found 20 Star Wars sets

4. Creating demo user...
👤 Created user with ID: [user_id]

5. Rating some sets...
⭐ Rated [set_name] with 5 stars

6. Getting personalized recommendations...
🎯 Got 5 recommendations:
   - [Set Name] (Score: 1.00)
     Reasons: Same theme, Similar complexity, etc.

7. Adding recommendation to wishlist...
💝 Added [set_name] to wishlist

8. Getting API metrics...
📊 API Metrics: [performance data]

✅ Demo completed!
```

## API Endpoints Demonstrated

The example client demonstrates these key API endpoints:

- `GET /health` - Health check
- `GET /themes` - List all LEGO themes
- `POST /search/sets` - Search for sets
- `POST /users` - Create new user
- `POST /users/{user_id}/interactions` - Rate sets
- `POST /recommendations` - Get personalized recommendations
- `POST /users/{user_id}/wishlist` - Manage wishlist
- `GET /metrics` - API performance metrics

## Customization

You can modify `example_client.py` to:

1. **Test different user behaviors**:
   ```python
   # Rate different types of sets
   rating_data = {"set_num": "10242-1", "rating": 4}
   ```

2. **Try different search queries**:
   ```python
   search_payload = {"query": "castle", "min_pieces": 100}
   ```

3. **Request different recommendation types**:
   ```python
   rec_payload = {"recommendation_type": "content", "set_num": "75192-1"}
   ```

## Prerequisites

Before running examples:

1. **Database running**: `docker-compose up -d`
2. **Data loaded**: `./reset_db.sh`
3. **API running**: `python src/scripts/recommendation_api.py`

## Creating Your Own Examples

To create a new example:

1. **Copy the template**:
   ```python
   import requests
   import json
   
   BASE_URL = "http://localhost:8000"
   
   def your_example():
       # Your code here
       pass
   
   if __name__ == "__main__":
       your_example()
   ```

2. **Add error handling**:
   ```python
   try:
       response = requests.post(f"{BASE_URL}/endpoint", json=data)
       response.raise_for_status()
       result = response.json()
   except requests.exceptions.RequestException as e:
       print(f"Error: {e}")
   ```

3. **Test your example**:
   ```bash
   python examples/your_example.py
   ```

## Integration with Frontend

The example client can serve as a reference for frontend integration:

- **React/Vue.js**: Use the same API calls with axios/fetch
- **Mobile apps**: Adapt the HTTP requests to your platform
- **Jupyter notebooks**: Use for data analysis and exploration

## Troubleshooting

### Common Issues
- **Connection refused**: Make sure API is running on port 8000
- **404 errors**: Check API endpoints are correctly formatted
- **Empty results**: Verify database has data loaded

### Debug Mode
Enable detailed logging in examples:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**🎯 Quick Start**: Run `python examples/example_client.py` after starting the API
**📚 Learn More**: Check the API documentation and test files
**🔧 Customize**: Modify examples for your specific use case
