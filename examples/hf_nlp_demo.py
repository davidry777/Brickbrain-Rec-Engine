#!/usr/bin/env python3
"""
HuggingFace NLP Demo Script

This script demonstrates the new HuggingFace-based NLP capabilities
for LEGO recommendations, replacing the Ollama dependency.
"""

import sys
import os
import requests
import json
from typing import Dict, Any, List
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

class HuggingFaceNLPDemo:
    """Demo class for HuggingFace NLP capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize demo with API base URL
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.user_id = "demo_user_001"
        
    def check_api_health(self) -> bool:
        """Check if the API is healthy and HuggingFace NLP is available"""
        try:
            response = self.session.get(f"{self.base_url}/health/detailed", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print("🏥 API Health Status:")
                print(f"   Overall Status: {health_data.get('status', 'unknown')}")
                print(f"   Database: {health_data['components'].get('database', 'unknown')}")
                print(f"   HuggingFace NLP: {health_data['components'].get('huggingface_nlp', 'unknown')}")
                
                if health_data['components'].get('huggingface_nlp') == 'healthy':
                    print("✅ HuggingFace NLP system is ready!")
                    return True
                else:
                    print("⚠️ HuggingFace NLP system is not fully available")
                    return False
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to API: {e}")
            return False
    
    def demo_natural_language_query(self):
        """Demonstrate natural language query processing"""
        print("\n" + "="*80)
        print("🧠 NATURAL LANGUAGE QUERY PROCESSING")
        print("="*80)
        
        test_queries = [
            "Star Wars sets for adults with lots of detail",
            "Medieval castle with minifigures under $200",
            "Technic sets for advanced builders",
            "Birthday gift for 10-year-old who loves space",
            "Something similar to Hogwarts Castle but smaller"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 60)
            
            try:
                response = self.session.post(
                    f"{self.base_url}/nlp/query",
                    json={
                        "query": query,
                        "user_id": self.user_id,
                        "top_k": 5,
                        "use_context": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    print(f"   Intent: {data.get('intent', 'N/A')}")
                    print(f"   Confidence: {data.get('confidence', 0):.2f}")
                    
                    if data.get('entities'):
                        print(f"   Entities: {data['entities']}")
                    
                    if data.get('filters'):
                        print(f"   Filters: {data['filters']}")
                    
                    if data.get('recommendations'):
                        print(f"   Found {len(data['recommendations'])} recommendations:")
                        for rec in data['recommendations'][:3]:
                            print(f"     • {rec.get('name', 'Unknown')} ({rec.get('set_num', 'N/A')})")
                            print(f"       {rec.get('theme', 'N/A')} | {rec.get('num_parts', 'N/A')} pieces")
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
            
            time.sleep(1)  # Brief pause between requests
    
    def demo_conversational_ai(self):
        """Demonstrate conversational AI with memory"""
        print("\n" + "="*80)
        print("💬 CONVERSATIONAL AI WITH MEMORY")
        print("="*80)
        
        conversation_flow = [
            "I need a birthday gift for my nephew who loves space themes",
            "Something around 500 pieces would be perfect",
            "Are there any with minifigures?",
            "What about similar sets but from different themes?",
            "Can you show me something more challenging?"
        ]
        
        session_id = None
        
        for i, message in enumerate(conversation_flow, 1):
            print(f"\n{i}. User: {message}")
            print("-" * 60)
            
            try:
                payload = {
                    "message": message,
                    "user_id": self.user_id,
                    "include_suggestions": True
                }
                
                if session_id:
                    payload["session_id"] = session_id
                
                response = self.session.post(
                    f"{self.base_url}/nlp/chat",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    session_id = data.get('session_id')
                    
                    print(f"   🤖 Assistant: {data.get('response', 'No response')}")
                    print(f"   Intent: {data.get('intent', 'N/A')} (confidence: {data.get('confidence', 0):.2f})")
                    
                    if data.get('recommendations'):
                        print(f"   📋 Recommendations ({len(data['recommendations'])}):")
                        for rec in data['recommendations'][:2]:
                            print(f"     • {rec.get('name', 'Unknown')} - {rec.get('theme', 'N/A')}")
                    
                    if data.get('follow_up_suggestions'):
                        print(f"   💭 Suggestions: {', '.join(data['follow_up_suggestions'][:2])}")
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
            
            time.sleep(1.5)  # Pause between conversation turns
    
    def demo_query_understanding(self):
        """Demonstrate advanced query understanding and explanation"""
        print("\n" + "="*80)
        print("🔍 QUERY UNDERSTANDING & EXPLANATION")
        print("="*80)
        
        complex_queries = [
            "I want a challenging Technic set with motors for my teenage son's birthday",
            "Find me affordable Creator sets under $50 for beginners",
            "Show me detailed architecture sets similar to the Statue of Liberty"
        ]
        
        for i, query in enumerate(complex_queries, 1):
            print(f"\n{i}. Analyzing: '{query}'")
            print("-" * 60)
            
            try:
                response = self.session.post(
                    f"{self.base_url}/nlp/understand",
                    json={
                        "query": query,
                        "include_explanation": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    print(f"   Intent: {data.get('intent', 'N/A')}")
                    print(f"   Confidence: {data.get('confidence', 0):.2f}")
                    print(f"   Semantic Query: '{data.get('semantic_query', 'N/A')}'")
                    
                    if data.get('entities'):
                        print("   Entities:")
                        for key, value in data['entities'].items():
                            print(f"     • {key}: {value}")
                    
                    if data.get('filters'):
                        print("   Filters:")
                        for key, value in data['filters'].items():
                            print(f"     • {key}: {value}")
                    
                    if data.get('explanation'):
                        print("   Explanation:")
                        for line in data['explanation'].split('\n\n'):
                            print(f"     {line}")
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
            
            time.sleep(1)
    
    def demo_conversation_memory(self):
        """Demonstrate conversation memory capabilities"""
        print("\n" + "="*80)
        print("🧠 CONVERSATION MEMORY DEMO")
        print("="*80)
        
        # First, have a conversation
        print("1. Building conversation memory...")
        session_id = None
        
        memory_building_queries = [
            "Show me Star Wars sets",
            "Something for advanced builders",
            "Under $150 please"
        ]
        
        for query in memory_building_queries:
            try:
                payload = {
                    "message": query,
                    "user_id": self.user_id,
                    "include_suggestions": False
                }
                if session_id:
                    payload["session_id"] = session_id
                
                response = self.session.post(f"{self.base_url}/nlp/chat", json=payload)
                if response.status_code == 200:
                    session_id = response.json().get('session_id')
                    print(f"   ✓ Added: '{query}'")
                
            except Exception as e:
                print(f"   ❌ Failed to add query: {e}")
        
        # Check memory
        if session_id:
            print(f"\n2. Retrieving conversation memory (Session: {session_id[:8]}...):")
            try:
                response = self.session.get(
                    f"{self.base_url}/nlp/memory/{self.user_id}",
                    params={"session_id": session_id}
                )
                
                if response.status_code == 200:
                    memory_data = response.json()
                    
                    print(f"   Total interactions: {memory_data.get('total_interactions', 0)}")
                    
                    if memory_data.get('conversation_history'):
                        print("   Recent conversation:")
                        for turn in memory_data['conversation_history'][-3:]:
                            print(f"     User: {turn['user_message']}")
                            print(f"     Assistant: {turn['assistant_response'][:50]}...")
                    
                    if memory_data.get('user_profile'):
                        profile = memory_data['user_profile']
                        print("   User profile:")
                        if profile.get('preferred_themes'):
                            print(f"     Preferred themes: {profile['preferred_themes']}")
                        print(f"     Interaction count: {profile.get('interaction_count', 0)}")
                else:
                    print(f"   ❌ Error retrieving memory: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Get conversation summary
        if session_id:
            print("\n3. Generating conversation summary:")
            try:
                response = self.session.get(f"{self.base_url}/nlp/summary/{self.user_id}/{session_id}")
                
                if response.status_code == 200:
                    summary_data = response.json()
                    print(f"   Summary: {summary_data.get('summary', 'No summary available')}")
                else:
                    print(f"   ❌ Error generating summary: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    def demo_user_feedback_learning(self):
        """Demonstrate user feedback and learning"""
        print("\n" + "="*80)
        print("📊 USER FEEDBACK & LEARNING")
        print("="*80)
        
        print("1. Recording user feedback...")
        
        # Simulate feedback on conversation turns
        try:
            feedback_data = {
                "user_id": self.user_id,
                "session_id": "demo_session",
                "turn_index": 0,
                "feedback": "liked",
                "rating": 5
            }
            
            response = self.session.post(f"{self.base_url}/nlp/feedback", json=feedback_data)
            
            if response.status_code == 200:
                print("   ✓ Positive feedback recorded")
            else:
                print(f"   ⚠️ Feedback recording: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        # Get user analytics
        print("\n2. Retrieving user analytics...")
        try:
            response = self.session.get(f"{self.base_url}/nlp/analytics/{self.user_id}")
            
            if response.status_code == 200:
                analytics = response.json()
                
                print("   User Analytics:")
                if analytics.get('personalization_context'):
                    context = analytics['personalization_context']
                    print(f"     Preference strength: {context.get('preference_strength', 'unknown')}")
                    if context.get('recent_themes'):
                        print(f"     Recent themes: {', '.join(context['recent_themes'][:3])}")
                
                if analytics.get('analytics'):
                    stats = analytics['analytics']
                    if stats.get('conversation_stats'):
                        conv_stats = stats['conversation_stats']
                        print(f"     Total conversations: {conv_stats.get('total_conversations', 0)}")
            else:
                print(f"   ⚠️ Analytics retrieval: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    def demo_migration_info(self):
        """Show migration information from Ollama to HuggingFace"""
        print("\n" + "="*80)
        print("🔄 MIGRATION FROM OLLAMA TO HUGGINGFACE")
        print("="*80)
        
        try:
            response = self.session.get(f"{self.base_url}/api/migration-guide")
            
            if response.status_code == 200:
                guide = response.json()['migration_guide']
                
                print("Migration Overview:")
                print(f"   {guide['overview']}")
                
                print("\nRecommended Endpoint Changes:")
                for feature, info in guide['recommended_endpoints'].items():
                    print(f"   {feature.title()}:")
                    print(f"     Old: {info['old']}")
                    print(f"     New: {info['new']}")
                    print(f"     Description: {info['description']}")
                
                print("\nNew Features:")
                for feature in guide['new_features'][:5]:
                    print(f"   • {feature}")
                
                print("\nConfiguration:")
                for var, desc in guide['configuration']['environment_variables'].items():
                    print(f"   {var}: {desc}")
            else:
                print(f"❌ Failed to get migration guide: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    def run_full_demo(self):
        """Run the complete HuggingFace NLP demo"""
        print("🧱 HuggingFace LEGO NLP Recommendation Demo")
        print("=" * 80)
        print("This demo showcases the new HuggingFace-based NLP capabilities")
        print("for LEGO recommendations, replacing the Ollama dependency.")
        print("=" * 80)
        
        # Check API health first
        if not self.check_api_health():
            print("\n❌ API is not available or HuggingFace NLP is not ready.")
            print("Please ensure the API server is running and HuggingFace models are loaded.")
            return
        
        try:
            # Run all demos
            self.demo_natural_language_query()
            self.demo_conversational_ai()
            self.demo_query_understanding()
            self.demo_conversation_memory()
            self.demo_user_feedback_learning()
            self.demo_migration_info()
            
            print("\n" + "="*80)
            print("🎉 HUGGINGFACE NLP DEMO COMPLETED!")
            print("="*80)
            print("\nKey Features Demonstrated:")
            print("✅ Enhanced natural language processing with HuggingFace models")
            print("✅ Advanced conversation memory with SQLite backend")
            print("✅ Improved intent classification and entity extraction")
            print("✅ User preference learning from feedback")
            print("✅ Contextual follow-up suggestions")
            print("✅ Conversation summarization capabilities")
            print("✅ Migration path from Ollama to HuggingFace")
            
            print("\nBenefits over Ollama:")
            print("• Better memory efficiency with model quantization")
            print("• No dependency on external Ollama service")
            print("• Improved accuracy with specialized models")
            print("• Enhanced conversation context handling")
            print("• Local inference with GPU/CPU optimization")
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HuggingFace NLP Demo for LEGO Recommendations")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--demo", choices=[
        "health", "query", "conversation", "understanding", 
        "memory", "feedback", "migration", "all"
    ], default="all", help="Specific demo to run")
    
    args = parser.parse_args()
    
    demo = HuggingFaceNLPDemo(args.url)
    
    if args.demo == "all":
        demo.run_full_demo()
    elif args.demo == "health":
        demo.check_api_health()
    elif args.demo == "query":
        demo.demo_natural_language_query()
    elif args.demo == "conversation":
        demo.demo_conversational_ai()
    elif args.demo == "understanding":
        demo.demo_query_understanding()
    elif args.demo == "memory":
        demo.demo_conversation_memory()
    elif args.demo == "feedback":
        demo.demo_user_feedback_learning()
    elif args.demo == "migration":
        demo.demo_migration_info()
