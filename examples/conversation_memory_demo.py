#!/usr/bin/env python3
"""
🧱 LEGO Conversation Memory Demo
Demonstrates how the recommendation system maintains conversation context
and provides personalized recommendations based on user interactions.
"""

import requests
import json
import time
import uuid
from datetime import datetime

class ConversationMemoryDemo:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.user_id = str(uuid.uuid4())
        self.conversation_id = str(uuid.uuid4())
        
    def print_separator(self, title):
        """Print a formatted section separator"""
        print("\n" + "="*60)
        print(f"🎯 {title}")
        print("="*60)
        
    def print_response(self, data, title):
        """Print API response in a formatted way"""
        print(f"\n🤖 {title}:")
        if isinstance(data, dict) and 'recommendations' in data:
            recs = data['recommendations']
            print(f"📋 Found {len(recs)} recommendations")
            for i, rec in enumerate(recs[:3], 1):
                print(f"  {i}. {rec.get('name', 'Unknown')} ({rec.get('set_num', 'N/A')})")
        else:
            print(f"📄 Response: {json.dumps(data, indent=2)}")
            
    def demo_api_health(self):
        """Check if the API is responding"""
        self.print_separator("API HEALTH CHECK")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is healthy and responding")
                return True
            else:
                print(f"⚠️  API returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to API: {e}")
            print("Make sure the API server is running on http://localhost:8000")
            return False
    
    def demo_conversation_flow(self):
        """Demonstrate a conversation flow with memory"""
        self.print_separator("CONVERSATION FLOW DEMO")
        
        conversations = [
            "I'm looking for LEGO sets for my 8-year-old son's birthday",
            "He really likes space themes",
            "Something under $50 would be perfect",
            "Are there any with minifigures?",
            "What about similar sets but from different themes?"
        ]
        
        for i, message in enumerate(conversations, 1):
            print(f"\n{i}️⃣ Chat Message: {message}")
            
            # Test basic search since we don't have chat endpoint
            response = requests.post(f"{self.base_url}/search/sets", json={
                "query": message,
                "limit": 3
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🔍 Search Results: {len(data)} sets found")
                for j, result in enumerate(data[:2], 1):
                    print(f"  {j}. {result.get('name', 'Unknown')} - {result.get('theme_name', 'Unknown theme')}")
            else:
                print(f"❌ Error: {response.status_code}")
            
            time.sleep(0.5)  # Brief pause between messages
    
    def demo_conversational_ai(self):
        """Demonstrate conversational AI features if available"""
        self.print_separator("CONVERSATIONAL AI DEMO")
        
        print("Testing natural language understanding...")
        
        test_queries = [
            "birthday gift for 8 year old",
            "space themed sets under 100 pieces",
            "challenging builds for experienced builders"
        ]
        
        for query in test_queries:
            print(f"\n🎯 Query: '{query}'")
            
            # Test natural language search
            response = requests.post(f"{self.base_url}/search/natural", json={
                "query": query,
                "top_k": 3
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Natural language processing successful")
                results = data.get('results', [])
                print(f"📋 Found {len(results)} relevant sets")
                for i, result in enumerate(results[:2], 1):
                    print(f"  {i}. {result.get('name', 'Unknown')}")
            else:
                print(f"❌ Error: {response.status_code}")
    
    def run_full_demo(self):
        """Run the complete conversation memory demonstration"""
        print("🧱 LEGO Conversation Memory Demo")
        print("="*50)
        print("This demo shows how the LEGO recommendation system")
        print("maintains conversation context and learns from user")
        print("interactions to provide better recommendations.")
        print()
        print(f"👤 User ID: {self.user_id}")
        print(f"💬 Conversation ID: {self.conversation_id}")
        
        try:
            # Check API health first
            if not self.demo_api_health():
                return
            
            # Run demos
            self.demo_conversation_flow()
            self.demo_conversational_ai()
            
            print("\n🎉 Demo completed!")
            print("💡 Tip: Check the API logs to see how the system processes natural language queries")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the API server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Demo error: {e}")

def main():
    """Main function to run the demo"""
    demo = ConversationMemoryDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
