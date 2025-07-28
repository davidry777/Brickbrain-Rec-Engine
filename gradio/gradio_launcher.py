#!/usr/bin/env python3
"""
Simple Gradio Launcher for Brickbrain LEGO Recommendation System

This is a simplified launcher that can be run directly on your host system
if you have gradio installed locally, or from within the Docker container.

Usage:
1. From host (if gradio installed): python3 gradio_launcher.py
2. From container: docker-compose exec app conda run -n brickbrain-rec python /app/gradio_launcher.py
"""

import sys
import os

try:
    import gradio as gr
    import requests
    print("✅ Gradio and requests are available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("💡 Install with: pip install gradio requests")
    sys.exit(1)

# Simple health check function
def check_api_health():
    """Simple health check for the API"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return "🟢 API is healthy and ready!"
        else:
            return f"🔴 API responded with status {response.status_code}"
    except Exception as e:
        return f"🔴 Cannot connect to API: {str(e)}"

# Simple search function
def natural_language_search(query):
    """Simple natural language search"""
    if not query.strip():
        return "Please enter a search query!"
    
    try:
        response = requests.post(
            "http://localhost:8000/search/natural",
            json={"query": query, "top_k": 5, "include_explanation": True},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            result_text = f"## Search Results for: '{query}'\n\n"
            
            if data.get('results'):
                for i, result in enumerate(data['results'], 1):
                    result_text += f"**{i}. {result['name']}** ({result['set_num']})\n"
                    result_text += f"   - Theme: {result['theme']} | Year: {result['year']} | Parts: {result['num_parts']}\n"
                    result_text += f"   - Score: {result.get('relevance_score', 0):.3f}\n\n"
            else:
                result_text += "No results found.\n"
            
            if data.get('explanation'):
                result_text += f"\n### AI Explanation:\n{data['explanation']}"
            
            return result_text
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create simplified Gradio interface
with gr.Blocks(title="🧱 Brickbrain LEGO Recommender - Simple Demo") as demo:
    
    gr.Markdown("""
    # 🧱 Brickbrain LEGO Recommendation System - Simple Demo
    
    This is a simplified demonstration of the Brickbrain system. 
    
    **Prerequisites**: Make sure your Docker Compose services are running:
    ```bash
    docker-compose up -d
    ```
    """)
    
    # Health Check Section
    with gr.Row():
        health_btn = gr.Button("🔍 Check API Health", variant="primary")
        health_status = gr.Textbox(label="Status", interactive=False)
    
    health_btn.click(fn=check_api_health, outputs=health_status)
    
    gr.Markdown("---")
    
    # Natural Language Search Section
    gr.Markdown("## 🔍 Natural Language Search")
    gr.Markdown("Search for LEGO sets using everyday language!")
    
    with gr.Row():
        search_input = gr.Textbox(
            label="Search Query",
            placeholder="e.g., 'Star Wars sets for adults with lots of pieces'",
            lines=2,
            scale=3
        )
        search_btn = gr.Button("🔍 Search", variant="primary", scale=1)
    
    search_output = gr.Markdown(label="Search Results")
    
    search_btn.click(
        fn=natural_language_search,
        inputs=search_input,
        outputs=search_output
    )
    
    # Example queries
    gr.Markdown("### 💡 Try These Examples:")
    
    examples = [
        "Star Wars sets with lots of pieces",
        "Birthday gift for 8 year old",
        "Technic sets for advanced builders",
        "Simple Creator sets for beginners",
        "Architecture sets for display"
    ]
    
    for example in examples:
        example_btn = gr.Button(f"'{example}'", size="sm")
        example_btn.click(
            fn=lambda ex=example: ex,
            outputs=search_input
        )
    
    gr.Markdown("""
    ---
    ### 🔗 Full API Documentation
    - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
    - **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
    
    ### 🚀 Next Steps
    For the full-featured interface with conversational AI, user management, and more:
    ```bash
    ./launch_gradio.sh
    ```
    Or run the full gradio_interface.py script.
    """)

if __name__ == "__main__":
    print("🧱 Starting Simple Brickbrain Gradio Demo...")
    print("🔍 Make sure Docker Compose is running: docker-compose up -d")
    print("🌐 Interface will be available at: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
