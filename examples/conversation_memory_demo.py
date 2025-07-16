#!/usr/bin/env python3"""Conversation Memory Demo ScriptThis script demonstrates the conversation memory capabilities of the LEGO NLP Recommender,showing how the system maintains context across multiple interactions and learns from user feedback."""import requestsimport jsonimport timefrom typing import Dict, Anyclass ConversationMemoryDemo:    def __init__(self, base_url: str = "http://localhost:8000"):        self.base_url = base_url        self.user_id = 123        self.conversation_id = "demo_session_001"            def print_separator(self, title: str):        """Print a formatted separator for demo sections"""        print("\n" + "="*80)        print(f"🧠 {title}")        print("="*80)        def print_response(self, response: Dict[Any, Any], title: str):        """Print formatted response"""        print(f"\n📤 {title}")        print("-" * 40)        print(json.dumps(response, indent=2))        def demo_conversation_flow(self):        """Demonstrate a complete conversation flow with memory"""        self.print_separator("CONVERSATION MEMORY DEMO")                # 1. Initial query        print("\n1️⃣ Initial Query: Setting up context")        response = requests.post(f"{self.base_url}/search/natural", json={            "query": "I'm looking for Star Wars sets for my 10-year-old nephew",            "user_id": self.user_id,            "use_context": True,            "top_k": 3        })                if response.status_code == 200:            self.print_response(response.json(), "Initial Star Wars Search")        else:            print(f"❌ Error: {response.status_code}")            return                # 2. Follow-up query using context        print("\n2️⃣ Follow-up Query: Using conversation context")        time.sleep(1)  # Brief pause for demonstration                response = requests.post(f"{self.base_url}/search/natural", json={            "query": "What about something with fewer pieces?",            "user_id": self.user_id,            "use_context": True,            "top_k": 3        })                if response.status_code == 200:            self.print_response(response.json(), "Context-Aware Follow-up")        else:            print(f"❌ Error: {response.status_code}")                # 3. Record user feedback        print("\n3️⃣ User Feedback: Learning preferences")                # Simulate user liking a Star Wars set        feedback_response = requests.post(f"{self.base_url}/nlp/conversation/feedback", json={            "user_id": self.user_id,            "set_num": "75309-1",  # Republic Gunship            "feedback": "liked",            "rating": 5        })                if feedback_response.status_code == 200:            print("✅ Feedback recorded: User liked Republic Gunship")        else:            print(f"❌ Feedback error: {feedback_response.status_code}")                # 4. Another query that should benefit from learned preferences        print("\n4️⃣ Preference-Enhanced Query: Leveraging learned preferences")        time.sleep(1)                response = requests.post(f"{self.base_url}/search/natural", json={            "query": "Show me another set",            "user_id": self.user_id,            "use_context": True,            "top_k": 3        })                if response.status_code == 200:            self.print_response(response.json(), "Preference-Enhanced Recommendations")        else:            print(f"❌ Error: {response.status_code}")                # 5. Get conversation context        print("\n5️⃣ Conversation Context: Viewing accumulated context")                context_response = requests.get(f"{self.base_url}/nlp/conversation/memory", params={            "user_id": self.user_id        })                if context_response.status_code == 200:            self.print_response(context_response.json(), "Conversation Context")        else:            print(f"❌ Context error: {context_response.status_code}")
    
    def demo_conversational_ai(self):
        """Demonstrate advanced conversational AI features"""
        self.print_separator("CONVERSATIONAL AI DEMO")
        
        # Conversational AI with memory
        conversations = [
            "I need a birthday gift for my nephew who loves space themes",
            "Something around 500 pieces would be perfect",
            "Are there any with minifigures?",
            "What about similar sets but from different themes?"
        ]
        
        for i, message in enumerate(conversations, 1):
            print(f"\n{i}️⃣ Chat Message: {message}")
            
            response = requests.post(f"{self.base_url}/ai/chat", json={
                "message": message,
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 AI Response: {data.get('response', 'No response')}")
                if 'recommendations' in data:
                    print(f"📋 Recommendations: {len(data['recommendations'])} sets")
            else:
                print(f"❌ Error: {response.status_code}")
            
            time.sleep(0.5)  # Brief pause between messages
    
    def demo_follow_up_handling(self):
        """Demonstrate follow-up query handling"""
        self.print_separator("FOLLOW-UP QUERY DEMO")
        
        # Initial recommendation
        print("\n1️⃣ Initial Recommendation Request")
        response = requests.post(f"{self.base_url}/recommendations/conversational", json={
            "query": "Technic sets for advanced builders",
            "user_id": self.user_id,
            "top_k": 3
        })
        
        if response.status_code == 200:
            self.print_response(response.json(), "Initial Technic Recommendations")
        else:
            print(f"❌ Error: {response.status_code}")
            return
        
        # Follow-up queries
        follow_ups = [
            "Show me similar sets to those",
            "What about something smaller?",
            "Are there any with motors?"
        ]
        
        for i, query in enumerate(follow_ups, 2):
            print(f"\n{i}️⃣ Follow-up: {query}")
            
            response = requests.post(f"{self.base_url}/ai/follow-up", json={
                "query": query,
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🎯 Follow-up Response: {data.get('explanation', 'No explanation')}")
                if 'results' in data:
                    print(f"📋 Results: {len(data['results'])} sets")
            else:
                print(f"❌ Error: {response.status_code}")
    
    def demo_memory_management(self):
        """Demonstrate memory management features"""
        self.print_separator("MEMORY MANAGEMENT DEMO")
        
        # Add multiple interactions to show memory accumulation
        print("\n1️⃣ Adding Multiple Interactions")
        
        queries = [
            "Show me Creator sets",
            "What about City themes?",
            "Something for beginners",
            "With lots of minifigures",
            "Under $100"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"   Adding query {i}: {query}")
            
            requests.post(f"{self.base_url}/search/natural", json={
                "query": query,
                "user_id": self.user_id,
                "use_context": True,
                "top_k": 1
            })
            
            time.sleep(0.2)
        
        # Check memory state
        print("\n2️⃣ Current Memory State")
        
        context_response = requests.get(f"{self.base_url}/nlp/conversation/memory", params={
            "user_id": self.user_id
        })
        
        if context_response.status_code == 200:
            data = context_response.json()
            print(f"📊 Previous searches: {len(data.get('previous_searches', []))}")
            print(f"🎯 User preferences: {data.get('user_preferences', {})}")
            print(f"📝 Conversation summary: {data.get('conversation_summary', 'None')}")
        
        # Clear memory
        print("\n3️⃣ Clearing Memory")
        
        clear_response = requests.post(f"{self.base_url}/nlp/conversation/memory/clear", json={
            "user_id": self.user_id
        })
        
        if clear_response.status_code == 200:
            print("✅ Memory cleared successfully")
        else:
            print(f"❌ Clear error: {clear_response.status_code}")
    
    def run_full_demo(self):
        """Run the complete conversation memory demonstration"""
        print("🧱 LEGO Conversation Memory Demo")
        print("=" * 50)
        print("This demo showcases the conversation memory capabilities")
        print("of the LEGO NLP Recommender system.")
        
        try:
            # Check if API is available
            health_response = requests.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                print(f"❌ API not available at {self.base_url}")
                return
            
            print("✅ API is available, starting demo...")
            
            # Run all demos
            self.demo_conversation_flow()
            self.demo_conversational_ai()
            self.demo_follow_up_handling()
            self.demo_memory_management()
            
            print("\n" + "="*80)
            print("🎉 CONVERSATION MEMORY DEMO COMPLETED!")
            print("="*80)
            print("\nKey Features Demonstrated:")
            print("✅ Context-aware conversation flow")
            print("✅ User preference learning from feedback")
            print("✅ Follow-up query understanding")
            print("✅ Conversational AI with memory")
            print("✅ Memory management and cleanup")
            print("\nThe system successfully maintains conversation context,")
            print("learns from user interactions, and provides increasingly")
            print("relevant recommendations over time.")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the API server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    demo = ConversationMemoryDemo()
    demo.run_full_demo()
