# 🧱 Brickbrain Gradio Interface

A beautiful, interactive web interface for the Brickbrain LEGO Recommendation System built with Gradio.

## 🌟 Features

### 🔍 **Natural Language Search**
- Search for LEGO sets using everyday language
- AI-powered query understanding with intent detection
- Rich result formatting with match explanations

### 💬 **Conversational AI**
- Chat naturally about LEGO recommendations
- Context-aware conversations with memory
- Follow-up question suggestions

### 🧠 **Query Understanding**
- See how the AI interprets your queries
- View extracted filters, entities, and intents
- Debug query processing in real-time

### 🔗 **Semantic Similarity**
- Find sets similar to specific LEGO sets
- Description-based similarity matching
- Advanced vector search capabilities

### 👤 **User Management**
- Create personalized user profiles
- Set preferences and favorite themes
- Get tailored recommendations

### 🔧 **System Monitoring**
- Real-time health checks
- Service status monitoring
- API connectivity verification

## 🚀 Quick Start

### Prerequisites

1. **Docker Compose Services Running**:
   ```bash
   docker-compose up -d
   ```

2. **Verify API is healthy**:
   ```bash
   curl http://localhost:8000/health
   ```

### Launch Options

#### Option 1: Simple Demo (Recommended for First Time)

If you have Python and pip installed locally:

```bash
# Install gradio locally
pip install gradio requests

# Run the simple demo (from project root)
python3 gradio/gradio_launcher.py
```

#### Option 2: Full Interface (Container)

Use the automated launch script:

```bash
# Make executable (first time only)
chmod +x scripts/launch_gradio.sh

# Launch full interface
./scripts/launch_gradio.sh
```

#### Option 3: Docker Service (Included in Main Compose)

The Gradio service is already included in the main docker-compose.yml:

```bash
# Start all services including Gradio
docker-compose up -d

# Or start only specific services
docker-compose up -d postgres app gradio
```

## 🌐 Access the Interface

Once launched, the Gradio interface will be available at:
- **Gradio Interface**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 📱 Interface Tabs

### 🔍 System Health
- Check if all services are running
- View system status and connectivity
- Monitor API health in real-time

### 🔍 Natural Language Search
- Enter queries like "Star Wars sets for adults"
- Get AI explanations for recommendations
- Try example queries to get started

### 💬 Conversational AI
- Chat naturally about LEGO sets
- Ask follow-up questions
- Maintain conversation context

### 🧠 Query Understanding
- Analyze how AI interprets queries
- View extracted filters and entities
- Debug query processing

### 🔗 Find Similar Sets
- Search for sets similar to specific models
- Use descriptions to find matches
- Semantic similarity search

### 👤 User Profile & Personalization
- Create user profiles with preferences
- Get personalized recommendations
- Choose recommendation algorithms

## 💡 Example Queries

### Natural Language Search
```
"Star Wars sets with lots of pieces for adults"
"Birthday gift for my 8 year old nephew who loves cars"
"Detailed Technic sets between 1000 and 2000 pieces"
"Simple Creator sets for beginners under $50"
"Architecture sets for display"
"Sets similar to the Millennium Falcon but smaller"
```

### Conversational Examples
```
"I'm looking for a gift for my nephew who loves space"
"What are some good sets for adult collectors?"
"Show me something similar to the Hogwarts Castle"
"I want to build something challenging this weekend"
"What's good for someone new to LEGO?"
"I need something under $100 for a birthday"
```

## 🔧 Troubleshooting

### Common Issues

**❌ "Cannot connect to API"**
```bash
# Check if services are running
docker-compose ps

# Start services if needed
docker-compose up -d

# Check API health
curl http://localhost:8000/health
```

**❌ "Gradio not found"**
```bash
# Option 1: Install locally
pip install gradio requests

# Option 2: Install in container
docker-compose exec app conda run -n brickbrain-rec pip install gradio
```

**❌ "Port 7860 already in use"**
```bash
# Find what's using the port
lsof -i :7860

# Kill the process or change port in the script
```

### Checking Logs

```bash
# View API logs
docker-compose logs app

# Follow logs in real-time
docker-compose logs -f app

# Check specific errors
docker-compose logs app | grep ERROR
```

## 🎨 Customization

### Changing the Port

Edit the launch scripts and change:
```python
demo.launch(server_port=7860)  # Change to your preferred port
```

### Adding Features

The interface is modular. You can:
1. Add new tabs in `gradio/gradio_interface.py`
2. Create new API interaction functions
3. Customize the UI theme and layout

### Environment Variables

Set these environment variables to customize behavior:
```bash
export BRICKBRAIN_API_URL="http://localhost:8000"  # Change API URL
export GRADIO_SERVER_PORT=7860  # Change Gradio port
export GRADIO_SHARE=false  # Set to true for public sharing
```

## 🔗 Integration with Main System

The Gradio interface connects to all major Brickbrain features:

- **Recommendation API** (`/recommendations`)
- **Natural Language Search** (`/search/natural`)
- **Query Understanding** (`/nlp/understand`)
- **Conversational AI** (`/recommendations/conversational`)
- **Semantic Search** (`/sets/similar/semantic`)
- **User Management** (`/users/*`)

## 🚀 Production Deployment

For production deployment:

1. **Security**: Enable authentication and HTTPS
2. **Scaling**: Use nginx reverse proxy
3. **Monitoring**: Add logging and metrics
4. **Backup**: Ensure conversation data persistence

Example nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📝 License

This interface is part of the Brickbrain LEGO Recommendation System project.

---

## 🤝 Contributing

To contribute to the Gradio interface:

1. Add new features in `gradio/gradio_interface.py`
2. Test with the simple launcher first
3. Update this README with new features
4. Submit a pull request

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the API documentation at http://localhost:8000/docs
3. Check Docker Compose logs: `docker-compose logs app`
4. Open an issue in the main repository

---

**Happy Building! 🧱**
