#!/usr/bin/env python3
"""
Gradio Interface for Brickbrain LEGO Recommendation System

This interface showcases the full capabilities of the LEGO recommendation system:
- Natural Language Search
- Conversational Recommendations 
- Query Understanding
- Semantic Similarity Search
- User Management
- System Health Monitoring

Prerequisites:
- Docker Compose services running (postgres + app)
- API available at http://localhost:8000
"""

import gradio as gr
import requests
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
import os

# Configuration
API_BASE_URL = os.getenv("BRICKBRAIN_API_URL", "http://localhost:8000")
DEFAULT_USER_ID = 1

class BrickbrainGradioInterface:
    def __init__(self):
        self.api_base = API_BASE_URL
        self.session = requests.Session()
        self.conversation_history = []
        self.current_user_id = DEFAULT_USER_ID
        
    def check_api_health(self) -> Tuple[str, str]:
        """Check if the API is healthy and ready"""
        try:
            response = self.session.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = "🟢 API Healthy"
                details = f"""
**System Status**: {data.get('status', 'OK')}
**Database**: {"🟢 Connected" if data.get('database_status') == 'connected' else "🔴 Issues"}
**NLP System**: {"🟢 Ready" if data.get('nlp_status') == 'ready' else "🔴 Not Ready"}
**Vector DB**: {"🟢 Initialized" if data.get('vectordb_status') == 'ready' else "🔴 Not Ready"}
**Uptime**: {data.get('uptime', 'Unknown')}
**Models**: {', '.join(data.get('available_models', []))}
                """
            else:
                status = "🔴 API Issues"
                details = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            status = "🔴 API Unavailable"
            details = f"Connection failed: {str(e)}"
        
        return status, details

    def natural_language_search(self, query: str, top_k: int = 5, include_explanation: bool = True) -> Tuple[str, str]:
        """Perform natural language search"""
        if not query.strip():
            return "Please enter a search query!", ""
            
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "user_id": self.current_user_id,
                "include_explanation": include_explanation
            }
            
            response = self.session.post(
                f"{self.api_base}/search/natural",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format results
                results_text = f"## 🔍 Search Results for: '{query}'\n\n"
                
                # Query understanding
                if 'query_understanding' in data:
                    qu = data['query_understanding']
                    results_text += f"**🧠 Understanding**: {qu.get('intent', 'search')} (confidence: {qu.get('confidence', 0):.2f})\n"
                    if qu.get('entities'):
                        results_text += f"**📋 Extracted Info**: {', '.join([f'{k}: {v}' for k, v in qu['entities'].items() if v])}\n"
                    results_text += "\n"
                
                # Results
                if data.get('results'):
                    results_text += f"### 🎯 Top {len(data['results'])} Matches:\n\n"
                    for i, result in enumerate(data['results'], 1):
                        results_text += f"**{i}. {result['name']} ({result['set_num']})**\n"
                        results_text += f"   - **Theme**: {result['theme']} | **Year**: {result['year']} | **Parts**: {result['num_parts']}\n"
                        results_text += f"   - **Score**: {result.get('relevance_score', 0):.3f}\n"
                        if result.get('match_reasons'):
                            results_text += f"   - **Why it matches**: {', '.join(result['match_reasons'])}\n"
                        results_text += "\n"
                else:
                    results_text += "No results found.\n"
                
                # Explanation
                explanation = ""
                if include_explanation and data.get('explanation'):
                    explanation = f"## 💡 Why These Recommendations?\n\n{data['explanation']}"
                
                return results_text, explanation
                
            else:
                return f"❌ Error: HTTP {response.status_code}", response.text
                
        except Exception as e:
            return f"❌ Error: {str(e)}", ""

    def conversational_chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """Handle conversational recommendations"""
        if not message.strip():
            return history, ""
            
        try:
            # Convert gradio history to our format
            conversation_history = []
            for human_msg, ai_msg in history:
                if human_msg:
                    conversation_history.append({"role": "human", "content": human_msg})
                if ai_msg:
                    conversation_history.append({"role": "assistant", "content": ai_msg})
            
            payload = {
                "query": message,
                "conversation_history": conversation_history,
                "context": {"user_id": self.current_user_id},
                "user_id": self.current_user_id
            }
            
            response = self.session.post(
                f"{self.api_base}/recommendations/conversational",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format AI response
                ai_response = ""
                
                if data.get('type') == 'gift_recommendations':
                    ai_response += "🎁 **Gift Recommendations**\n\n"
                elif data.get('type') == 'recommendations':
                    ai_response += "🔍 **LEGO Recommendations**\n\n"
                else:
                    ai_response += f"📝 **{data.get('type', 'Response').title()}**\n\n"
                
                # Add results
                if data.get('results'):
                    for i, result in enumerate(data['results'][:3], 1):  # Limit to top 3 for chat
                        ai_response += f"**{i}. {result['name']}** ({result['set_num']})\n"
                        ai_response += f"   {result['theme']} • {result['num_parts']} pieces • {result['year']}\n\n"
                
                # Add follow-up questions
                if data.get('follow_up_questions'):
                    ai_response += "\n💭 **You might also ask:**\n"
                    for q in data['follow_up_questions'][:2]:  # Limit to 2
                        ai_response += f"• {q}\n"
                
                # Update history
                history.append([message, ai_response])
                
                return history, ""
                
            else:
                error_response = f"❌ Sorry, I encountered an error: HTTP {response.status_code}"
                history.append([message, error_response])
                return history, ""
                
        except Exception as e:
            error_response = f"❌ Sorry, I encountered an error: {str(e)}"
            history.append([message, error_response])
            return history, ""

    def understand_query(self, query: str) -> str:
        """Analyze query understanding without search"""
        if not query.strip():
            return "Please enter a query to analyze!"
            
        try:
            response = self.session.post(
                f"{self.api_base}/nlp/understand",
                json={"query": query},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                analysis = f"## 🧠 Query Analysis for: '{query}'\n\n"
                analysis += f"**🎯 Intent**: {data.get('intent', 'unknown')}\n"
                analysis += f"**🎲 Confidence**: {data.get('confidence', 0):.2f}\n\n"
                
                if data.get('extracted_filters'):
                    analysis += "**🔍 Extracted Filters**:\n"
                    for key, value in data['extracted_filters'].items():
                        if value:
                            analysis += f"- {key}: {value}\n"
                    analysis += "\n"
                
                if data.get('extracted_entities'):
                    analysis += "**📋 Extracted Entities**:\n"
                    for key, value in data['extracted_entities'].items():
                        if value:
                            analysis += f"- {key}: {value}\n"
                    analysis += "\n"
                
                analysis += f"**🔤 Semantic Query**: {data.get('semantic_query', 'N/A')}\n\n"
                analysis += f"**💭 Interpretation**: {data.get('interpretation', 'No interpretation available')}\n"
                
                return analysis
                
            else:
                return f"❌ Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def find_similar_sets(self, set_query: str, description: str = "") -> str:
        """Find semantically similar sets"""
        if not set_query.strip():
            return "Please enter a set name or description!"
            
        # Check if this looks like a set number (contains hyphens and numbers)
        import re
        set_num_pattern = r'^\d+-\d+$'  # Matches patterns like "75192-1"
        
        if re.match(set_num_pattern, set_query.strip()):
            # Use the specific set similarity endpoint
            try:
                payload = {
                    "set_num": set_query.strip(),
                    "description": description if description.strip() else None,
                    "top_k": 5
                }
                
                response = self.session.post(
                    f"{self.api_base}/sets/similar/semantic",
                    json=payload,
                    timeout=20
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if results:
                        output = f"## 🔗 Sets Similar to: '{set_query}'\n\n"
                        for i, result in enumerate(results, 1):
                            output += f"**{i}. {result['name']} ({result['set_num']})**\n"
                            output += f"   - {result['theme']} | {result['year']} | {result['num_parts']} pieces\n"
                            output += f"   - Similarity: {result.get('score', 0):.3f}\n"
                            if result.get('description'):
                                output += f"   - *{result['description']}*\n"
                            output += "\n"
                        return output
                    else:
                        return "No similar sets found."
                        
                else:
                    return f"❌ Error: HTTP {response.status_code} - {response.text}"
                    
            except Exception as e:
                return f"❌ Error: {str(e)}"
        else:
            # Use natural language search for text descriptions
            full_query = f"Sets similar to {set_query}"
            if description.strip():
                full_query += f" {description}"
            
            try:
                payload = {
                    "query": full_query,
                    "top_k": 5,
                    "include_explanation": False
                }
                
                response = self.session.post(
                    f"{self.api_base}/search/natural",
                    json=payload,
                    timeout=20
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if results:
                        output = f"## 🔗 Sets Similar to: '{set_query}'\n\n"
                        for i, result in enumerate(results, 1):
                            output += f"**{i}. {result['name']} ({result['set_num']})**\n"
                            output += f"   - {result['theme']} | {result['year']} | {result['num_parts']} pieces\n"
                            output += f"   - Relevance: {result.get('relevance_score', 0):.3f}\n"
                            if result.get('match_reasons'):
                                output += f"   - Why: {', '.join(result['match_reasons'])}\n"
                            output += "\n"
                        return output
                    else:
                        return "No similar sets found."
                        
                else:
                    return f"❌ Error: HTTP {response.status_code} - {response.text}"
                    
            except Exception as e:
                return f"❌ Error: {str(e)}"

    def get_user_recommendations(self, algorithm: str = "hybrid", top_k: int = 5) -> str:
        """Get personalized recommendations for the current user"""
        try:
            payload = {
                "user_id": self.current_user_id,
                "top_k": top_k,
                "algorithm": algorithm
            }
            
            response = self.session.post(
                f"{self.api_base}/recommendations",
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                results = response.json()
                
                if results:
                    output = f"## 👤 Personalized Recommendations (User {self.current_user_id})\n"
                    output += f"**Algorithm**: {algorithm} | **Results**: {len(results)}\n\n"
                    
                    for i, result in enumerate(results, 1):
                        output += f"**{i}. {result['name']} ({result['set_num']})**\n"
                        output += f"   - {result['theme']} | {result['year']} | {result['num_parts']} pieces\n"
                        output += f"   - Score: {result.get('score', 0):.3f}\n"
                        if result.get('reasoning'):
                            output += f"   - Why: {result['reasoning']}\n"
                        output += "\n"
                    return output
                else:
                    return "No personalized recommendations available. The user may need more interaction history."
                    
            else:
                return f"❌ Error: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def create_user_profile(self, user_name: str, age: int, favorite_themes: str, experience_level: str) -> str:
        """Create or update user profile"""
        if not user_name.strip():
            return "Please enter a user name!"
            
        try:
            # Create user
            user_payload = {
                "name": user_name,
                "email": f"{user_name.lower().replace(' ', '_')}@example.com",
                "age": age
            }
            
            response = self.session.post(
                f"{self.api_base}/users",
                json=user_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                new_user_id = user_data.get('user_id')
                
                # Update preferences
                themes_list = [theme.strip() for theme in favorite_themes.split(',') if theme.strip()]
                prefs_payload = {
                    "favorite_themes": themes_list,
                    "experience_level": experience_level,
                    "age_group": "child" if age < 13 else "teen" if age < 18 else "adult"
                }
                
                prefs_response = self.session.put(
                    f"{self.api_base}/users/{new_user_id}/preferences",
                    json=prefs_payload,
                    timeout=10
                )
                
                self.current_user_id = new_user_id
                
                return f"""✅ **User Profile Created Successfully!**

**User ID**: {new_user_id}
**Name**: {user_name}
**Age**: {age}
**Experience**: {experience_level}
**Favorite Themes**: {', '.join(themes_list) if themes_list else 'None specified'}

This user ID ({new_user_id}) is now active for personalized recommendations."""
                
            else:
                return f"❌ Error creating user: HTTP {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Initialize the interface
interface = BrickbrainGradioInterface()

# Define example queries for natural language search
example_queries = [
    "Star Wars sets with lots of pieces for adults",
    "Birthday gift for my 8 year old nephew who loves cars",
    "Detailed Technic sets between 1000 and 2000 pieces",
    "Simple Creator sets for beginners under $50",
    "Architecture sets for display",
    "Sets similar to the Millennium Falcon but smaller"
]

# Create the Gradio interface
with gr.Blocks(title="🧱 Brickbrain LEGO Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧱 Brickbrain LEGO Recommendation System
    
    Welcome to the interactive demo of the Brickbrain LEGO Recommendation System! This showcases the power of AI-driven LEGO set recommendations using natural language processing, machine learning, and conversational AI.
    
    ## 🚀 Features Demonstrated:
    - **Natural Language Search**: Find LEGO sets using everyday language
    - **Conversational AI**: Chat naturally about LEGO recommendations
    - **Query Understanding**: See how AI interprets your requests
    - **Semantic Similarity**: Find similar sets based on descriptions
    - **Personalized Recommendations**: Get recommendations tailored to user preferences
    - **System Health**: Monitor the AI system status
    """)
    
    # System Health Tab
    with gr.Tab("🔍 System Health"):
        gr.Markdown("### Check if the Brickbrain API and all services are running properly")
        
        with gr.Row():
            health_check_btn = gr.Button("🔄 Check System Health", variant="primary")
        
        with gr.Row():
            health_status = gr.Textbox(label="System Status", interactive=False)
            health_details = gr.Markdown(label="System Details")
        
        health_check_btn.click(
            fn=interface.check_api_health,
            outputs=[health_status, health_details]
        )
    
    # Natural Language Search Tab
    with gr.Tab("🔍 Natural Language Search"):
        gr.Markdown("### Search for LEGO sets using natural language")
        
        with gr.Row():
            with gr.Column(scale=3):
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., 'Star Wars sets for adults with lots of pieces'",
                    lines=2
                )
            with gr.Column(scale=1):
                search_top_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of Results"
                )
                search_explanation = gr.Checkbox(
                    label="Include AI Explanation", value=True
                )
        
        with gr.Row():
            search_btn = gr.Button("🔍 Search", variant="primary")
            
        with gr.Row():
            search_results = gr.Markdown(label="Search Results")
            search_explanation_output = gr.Markdown(label="AI Explanation")
        
        # Example buttons
        gr.Markdown("### 💡 Try These Examples:")
        with gr.Row():
            for i in range(0, len(example_queries), 2):
                with gr.Column():
                    for j in range(2):
                        if i + j < len(example_queries):
                            example_btn = gr.Button(f"'{example_queries[i + j]}'", size="sm")
                            example_btn.click(
                                fn=lambda query=example_queries[i + j]: query,
                                outputs=search_query
                            )
        
        search_btn.click(
            fn=interface.natural_language_search,
            inputs=[search_query, search_top_k, search_explanation],
            outputs=[search_results, search_explanation_output]
        )
    
    # Conversational AI Tab
    with gr.Tab("💬 Conversational AI"):
        gr.Markdown("### Chat naturally about LEGO recommendations with AI memory")
        
        chatbot = gr.Chatbot(
            label="LEGO Recommendation Chat",
            height=400,
            placeholder="Start chatting about LEGO sets...",
            type='messages'
        )
        
        with gr.Row():
            chat_input = gr.Textbox(
                label="Your Message",
                placeholder="e.g., 'I'm looking for a challenging build for the weekend'",
                scale=4
            )
            chat_send = gr.Button("💬 Send", variant="primary", scale=1)
        
        with gr.Row():
            chat_clear = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        chat_send.click(
            fn=interface.conversational_chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            fn=interface.conversational_chat,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_clear.click(lambda: ([], ""), outputs=[chatbot, chat_input])
        
        # Example conversation starters
        gr.Markdown("### 💡 Try These Conversation Starters:")
        conversation_examples = [
            "I'm looking for a gift for my nephew who loves space",
            "What are some good sets for adult collectors?",
            "Show me something similar to the Hogwarts Castle",
            "I want to build something challenging this weekend",
            "What's good for someone new to LEGO?",
            "I need something under $100 for a birthday"
        ]
        
        with gr.Row():
            for i in range(0, len(conversation_examples), 2):
                with gr.Column():
                    for j in range(2):
                        if i + j < len(conversation_examples):
                            conv_example_btn = gr.Button(f"'{conversation_examples[i + j]}'", size="sm")
                            conv_example_btn.click(
                                fn=lambda msg=conversation_examples[i + j]: msg,
                                outputs=chat_input
                            )
    
    # Query Understanding Tab
    with gr.Tab("🧠 Query Understanding"):
        gr.Markdown("### See how the AI understands and interprets your queries")
        
        with gr.Row():
            understand_query = gr.Textbox(
                label="Query to Analyze",
                placeholder="e.g., 'birthday gift for 10 year old who likes cars'",
                lines=2
            )
            understand_btn = gr.Button("🧠 Analyze Query", variant="primary")
        
        understand_output = gr.Markdown(label="Query Analysis")
        
        understand_btn.click(
            fn=interface.understand_query,
            inputs=understand_query,
            outputs=understand_output
        )
    
    # Semantic Similarity Tab
    with gr.Tab("🔗 Find Similar Sets"):
        gr.Markdown("### Find LEGO sets similar to a given set or description")
        
        with gr.Row():
            with gr.Column():
                similar_query = gr.Textbox(
                    label="Set Name or Description",
                    placeholder="e.g., 'Millennium Falcon' or 'large detailed castle'",
                    lines=2
                )
                similar_description = gr.Textbox(
                    label="Additional Context (Optional)",
                    placeholder="e.g., 'but smaller and less expensive'",
                    lines=1
                )
            with gr.Column():
                similar_btn = gr.Button("🔗 Find Similar Sets", variant="primary")
        
        similar_results = gr.Markdown(label="Similar Sets")
        
        similar_btn.click(
            fn=interface.find_similar_sets,
            inputs=[similar_query, similar_description],
            outputs=similar_results
        )
    
    # User Profile & Personalization Tab
    with gr.Tab("👤 User Profile & Personalization"):
        gr.Markdown("### Create user profiles for personalized recommendations")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Create User Profile")
                user_name = gr.Textbox(label="Name", placeholder="Enter user name")
                user_age = gr.Slider(minimum=1, maximum=100, value=25, label="Age")
                user_themes = gr.Textbox(
                    label="Favorite Themes (comma-separated)", 
                    placeholder="e.g., Star Wars, Creator, Technic"
                )
                user_experience = gr.Dropdown(
                    choices=["beginner", "intermediate", "advanced", "expert"],
                    value="intermediate",
                    label="Experience Level"
                )
                create_profile_btn = gr.Button("👤 Create Profile", variant="primary")
                
            with gr.Column():
                gr.Markdown("#### Get Personalized Recommendations")
                rec_algorithm = gr.Dropdown(
                    choices=["hybrid", "content_based", "collaborative", "ml_enhanced"],
                    value="hybrid",
                    label="Recommendation Algorithm"
                )
                rec_count = gr.Slider(minimum=1, maximum=10, value=5, label="Number of Recommendations")
                get_recs_btn = gr.Button("🎯 Get Recommendations", variant="primary")
        
        with gr.Row():
            profile_output = gr.Markdown(label="Profile Creation Result")
            recommendations_output = gr.Markdown(label="Personalized Recommendations")
        
        create_profile_btn.click(
            fn=interface.create_user_profile,
            inputs=[user_name, user_age, user_themes, user_experience],
            outputs=profile_output
        )
        
        get_recs_btn.click(
            fn=interface.get_user_recommendations,
            inputs=[rec_algorithm, rec_count],
            outputs=recommendations_output
        )
    
    # Footer
    gr.Markdown("""
    ---
    ### 🔧 Technical Details
    
    This demo showcases a production-ready LEGO recommendation system built with:
    - **FastAPI** backend with **PostgreSQL** database
    - **LangChain** for natural language processing
    - **Local LLM** (Ollama with Mistral) for query understanding
    - **FAISS** vector database for semantic search
    - **Hybrid ML** recommendation algorithms
    - **Docker Compose** for easy deployment
    
    **System Requirements**: Docker Compose services must be running on `localhost:8000`
    
    💡 **Tip**: Check the System Health tab first to ensure all services are running!
    """)

# Launch the interface
if __name__ == "__main__":
    print("🧱 Starting Brickbrain Gradio Interface...")
    print("🔍 Make sure your Docker Compose services are running:")
    print("   docker-compose up -d")
    print("📡 API should be available at http://localhost:8000")
    print("🌐 Gradio will be available at http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        quiet=False
    )
