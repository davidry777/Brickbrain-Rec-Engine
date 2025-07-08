#!/usr/bin/env python3
"""
Natural Language LEGO Recommendation Demo

This script demonstrates the natural language capabilities of the Brickbrain system.
"""

import requests
import json
from typing import Dict, List
import time

# API base URL
BASE_URL = "http://localhost:8000"

class BrickbrainNLDemo:
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
    
    def check_health(self):
        """Check if the API is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            print("✅ API is healthy")
            return True
        except:
            print("❌ API is not running. Please start the API first.")
            return False
    
    def natural_language_search(self, query: str, user_id: int = None):
        """Perform natural language search"""
        print(f"\n🔍 Natural Language Search: '{query}'")
        print("-" * 80)
        
        payload = {
            "query": query,
            "user_id": user_id,
            "top_k": 5,
            "include_explanation": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/search/natural",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Display query understanding
            print(f"📊 Query Understanding:")
            print(f"   Intent: {data['query_understanding']['intent']}")
            print(f"   Confidence: {data['query_understanding']['confidence']:.2%}")
            
            if data['extracted_filters']:
                print(f"   Extracted Filters:")
                for key, value in data['extracted_filters'].items():
                    print(f"      - {key}: {value}")
            
            if data['query_understanding']['entities']:
                print(f"   Extracted Entities:")
                for key, value in data['query_understanding']['entities'].items():
                    print(f"      - {key}: {value}")
            
            # Display results
            print(f"\n🎯 Found {len(data['results'])} matching sets:")
            for i, result in enumerate(data['results'], 1):
                print(f"\n{i}. {result['name']} ({result['set_num']})")
                print(f"   Theme: {result['theme']} | Year: {result['year']} | Pieces: {result['num_parts']}")
                print(f"   Relevance: {result['relevance_score']:.2%}")
                print(f"   Match Reasons:")
                for reason in result['match_reasons']:
                    print(f"      ✓ {reason}")
            
            # Display explanation
            if data.get('explanation'):
                print(f"\n💡 Explanation:")
                print(f"   {data['explanation']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def find_similar_sets_semantic(self, set_num: str, context: str = None):
        """Find semantically similar sets"""
        print(f"\n🔄 Finding sets similar to {set_num}")
        if context:
            print(f"   Additional context: '{context}'")
        print("-" * 80)
        
        payload = {
            "set_num": set_num,
            "description": context,
            "top_k": 5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/sets/similar/semantic",
                json=payload
            )
            response.raise_for_status()
            results = response.json()
            
            print(f"Found {len(results)} similar sets:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['name']} ({result['set_num']})")
                print(f"   Theme: {result['theme']} | Year: {result['year']} | Pieces: {result['num_parts']}")
                print(f"   Similarity: {result['relevance_score']:.2%}")
                for reason in result['match_reasons']:
                    print(f"   ✓ {reason}")
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def conversational_recommendation(self, query: str, history: List[Dict] = None):
        """Get conversational recommendations"""
        print(f"\n💬 Conversational Query: '{query}'")
        print("-" * 80)
        
        payload = {
            "query": query,
            "conversation_history": history or [],
            "context": {}
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/recommendations/conversational",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"Response Type: {data['type']}")
            
            if 'results' in data:
                print(f"\n🎁 Recommendations:")
                for i, result in enumerate(data['results'], 1):
                    print(f"{i}. {result['name']} ({result['set_num']})")
                    print(f"   {result['theme']} | {result['num_parts']} pieces")
            
            if 'follow_up_questions' in data:
                print(f"\n❓ Follow-up Questions:")
                for q in data['follow_up_questions']:
                    print(f"   • {q}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def understand_query(self, query: str):
        """Analyze query understanding without search"""
        print(f"\n🧠 Understanding Query: '{query}'")
        print("-" * 80)
        
        try:
            response = self.session.post(
                f"{self.base_url}/nlp/understand",
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"Intent: {data['intent']} (confidence: {data['confidence']:.2%})")
            print(f"Interpretation: {data['interpretation']}")
            
            if data['extracted_filters']:
                print(f"\nExtracted Filters:")
                for key, value in data['extracted_filters'].items():
                    print(f"   {key}: {value}")
            
            if data['extracted_entities']:
                print(f"\nExtracted Entities:")
                for key, value in data['extracted_entities'].items():
                    print(f"   {key}: {value}")
            
            print(f"\nSemantic Query: '{data['semantic_query']}'")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_demo_scenarios(self):
        """Run various demo scenarios"""
        print("\n" + "="*80)
        print("🧱 BRICKBRAIN NATURAL LANGUAGE DEMO")
        print("="*80)
        
        # Scenario 1: Detailed search query
        self.natural_language_search(
            "I want a space station with lots of detail and over 1000 pieces"
        )
        time.sleep(1)
        
        # Scenario 2: Gift recommendation
        self.natural_language_search(
            "Looking for a birthday gift for my 8-year-old nephew who loves cars and racing"
        )
        time.sleep(1)
        
        # Scenario 3: Budget-conscious search
        self.natural_language_search(
            "Show me Star Wars sets under $100 that are good for beginners"
        )
        time.sleep(1)
        
        # Scenario 4: Complex requirements
        self.natural_language_search(
            "I need a challenging Technic set with motors, between 1500-3000 pieces"
        )
        time.sleep(1)
        
        # Scenario 5: Semantic similarity with context
        self.find_similar_sets_semantic(
            "75192-1",  # Millennium Falcon
            "but smaller and more affordable for a kid"
        )
        time.sleep(1)
        
        # Scenario 6: Collection advice
        conversation_history = [
            {"role": "user", "content": "I'm starting a LEGO architecture collection"},
            {"role": "assistant", "content": "Great choice! Architecture sets are wonderful for display."}
        ]
        self.conversational_recommendation(
            "What are some must-have sets for a beginner?",
            conversation_history
        )
        time.sleep(1)
        
        # Scenario 7: Query understanding
        queries_to_understand = [
            "vintage castle sets from the 90s",
            "something educational for STEM learning",
            "sets with light-up features and sound effects"
        ]
        
        for query in queries_to_understand:
            self.understand_query(query)
            time.sleep(1)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*80)
        print("🧱 BRICKBRAIN INTERACTIVE MODE")
        print("="*80)
        print("Type your LEGO-related queries in natural language!")
        print("Commands: 'quit' to exit, 'demo' to run demo scenarios")
        print("-"*80)
        
        while True:
            query = input("\n🎯 Your query: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Thanks for using Brickbrain!")
                break
            elif query.lower() == 'demo':
                self.run_demo_scenarios()
            elif query.startswith("similar to "):
                # Extract set number
                parts = query.split()
                if len(parts) >= 3:
                    set_num = parts[2]
                    context = " ".join(parts[3:]) if len(parts) > 3 else None
                    self.find_similar_sets_semantic(set_num, context)
            else:
                self.natural_language_search(query)


def main():
    """Main demo function"""
    demo = BrickbrainNLDemo()
    
    # Check if API is running
    if not demo.check_health():
        print("\nPlease start the API with: python src/scripts/recommendation_api.py")
        return
    
    print("\nWelcome to the Brickbrain Natural Language Demo!")
    print("\nOptions:")
    print("1. Run demo scenarios")
    print("2. Interactive mode")
    print("3. Quit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        demo.run_demo_scenarios()
    elif choice == "2":
        demo.interactive_mode()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()