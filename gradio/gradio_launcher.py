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

# Create custom theme with yellow background, red accents, and Futura font
custom_theme = gr.themes.Soft().set(
    # Background colors
    background_fill_primary="#FFEB3B",  # Yellow background
    background_fill_secondary="#FFF9C4",  # Lighter yellow for secondary areas
    
    # Button and accent colors (change purple to red)
    button_primary_background_fill="#F44336",  # Red primary buttons
    button_primary_background_fill_hover="#D32F2F",  # Darker red on hover
    button_primary_border_color="#F44336",
    button_primary_text_color="white",
    
    button_secondary_background_fill="#FFCDD2",  # Light red for secondary buttons
    button_secondary_background_fill_hover="#FFAB91",  # Orange-red on hover
    button_secondary_border_color="#F44336",
    button_secondary_text_color="#D32F2F",
    
    # Slider and interactive elements
    slider_color="#F44336",  # Red sliders
    checkbox_background_color="#F44336",  # Red checkboxes
    
    # Input field styling
    input_background_fill="#FFFDE7",  # Very light yellow for inputs
    input_border_color="#F57F17",  # Darker yellow border
    
    # Tab styling
    button_cancel_background_fill="#FFCDD2",  # Light red for inactive tabs
    button_cancel_text_color="#D32F2F",
)

# Add custom CSS for additional styling including Futura font
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Futura:wght@400;500;700&display=swap');

/* Apply Futura font globally */
* {
    font-family: 'Futura', Arial, sans-serif !important;
}

/* Yellow background for main container */
.gradio-container {
    background-color: #FFEB3B !important;
}

/* Override Gradio's default fonts */
.gradio-container * {
    font-family: 'Futura', Arial, sans-serif !important;
}

/* Ensure buttons use Futura */
button {
    font-family: 'Futura', Arial, sans-serif !important;
    font-weight: 500 !important;
}

/* Ensure text inputs use Futura */
input, textarea, select {
    font-family: 'Futura', Arial, sans-serif !important;
}

/* Style markdown headings with red color */
.markdown h1, .markdown h2, .markdown h3, .markdown h4, .markdown h5, .markdown h6 {
    color: #D32F2F !important;
    font-family: 'Futura', Arial, sans-serif !important;
    font-weight: 700 !important;
}

/* Style textboxes and text areas */
.textbox, .textarea {
    background-color: #FFFDE7 !important;
    border: 1px solid #F57F17 !important;
}

/* Custom styling for small buttons */
.btn-sm {
    background-color: #FFCDD2 !important;
    color: #D32F2F !important;
    border: 1px solid #F44336 !important;
    font-family: 'Futura', Arial, sans-serif !important;
}

.btn-sm:hover {
    background-color: #F44336 !important;
    color: white !important;
}

/* Style the main title */
.main-title {
    color: #D32F2F !important;
    font-weight: 700 !important;
}
"""

# Create simplified Gradio interface
with gr.Blocks(title="🧱 Brickbrain LEGO Recommender - Simple Demo", theme=custom_theme, css=custom_css) as demo:
    
    gr.Markdown("""
    <div class="main-title">
    
    # 🧱 Brickbrain LEGO Recommendation System - Simple Demo
    
    </div>
    
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
