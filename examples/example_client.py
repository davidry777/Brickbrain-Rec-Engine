#!/usr/bin/env python3
"""
Example client for the LEGO Recommendation API
Demonstrates how to interact with the API endpoints
"""

import requests
import json
from typing import Dict, List, Optional

class LegoAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_user(self, username: str, email: str, password: str) -> Dict:
        """Create a new user"""
        data = {
            "username": username,
            "email": email,
            "password": password
        }
        response = self.session.post(f"{self.base_url}/users", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get user profile"""
        response = self.session.get(f"{self.base_url}/users/{user_id}/profile")
        response.raise_for_status()
        return response.json()
    
    def track_interaction(self, user_id: int, set_num: str, interaction_type: str, 
                         rating: Optional[int] = None, source: str = "api") -> Dict:
        """Track a user interaction"""
        data = {
            "user_id": user_id,
            "set_num": set_num,
            "interaction_type": interaction_type,
            "source": source
        }
        if rating:
            data["rating"] = rating
        
        response = self.session.post(f"{self.base_url}/users/{user_id}/interactions", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_recommendations(self, user_id: Optional[int] = None, set_num: Optional[str] = None,
                          top_k: int = 10, recommendation_type: str = "hybrid") -> List[Dict]:
        """Get recommendations"""
        data = {
            "top_k": top_k,
            "recommendation_type": recommendation_type,
            "include_reasons": True
        }
        if user_id:
            data["user_id"] = user_id
        if set_num:
            data["set_num"] = set_num
        
        response = self.session.post(f"{self.base_url}/recommendations", json=data)
        response.raise_for_status()
        return response.json()
    
    def search_sets(self, query: Optional[str] = None, theme_ids: Optional[List[int]] = None,
                   min_pieces: Optional[int] = None, max_pieces: Optional[int] = None,
                   limit: int = 20) -> List[Dict]:
        """Search for sets"""
        data = {
            "limit": limit,
            "sort_by": "name",
            "sort_order": "asc"
        }
        if query:
            data["query"] = query
        if theme_ids:
            data["theme_ids"] = theme_ids
        if min_pieces:
            data["min_pieces"] = min_pieces
        if max_pieces:
            data["max_pieces"] = max_pieces
        
        response = self.session.post(f"{self.base_url}/search/sets", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_set_details(self, set_num: str, user_id: Optional[int] = None) -> Dict:
        """Get detailed information about a set"""
        params = {}
        if user_id:
            params["user_id"] = user_id
        
        response = self.session.get(f"{self.base_url}/sets/{set_num}", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_themes(self) -> List[Dict]:
        """Get all available themes"""
        response = self.session.get(f"{self.base_url}/themes")
        response.raise_for_status()
        return response.json()
    
    def add_to_collection(self, user_id: int, set_num: str) -> Dict:
        """Add a set to user's collection"""
        params = {"set_num": set_num}
        response = self.session.post(f"{self.base_url}/users/{user_id}/collection", params=params)
        response.raise_for_status()
        return response.json()
    
    def add_to_wishlist(self, user_id: int, set_num: str, priority: int = 3, 
                       max_price: Optional[float] = None) -> Dict:
        """Add a set to user's wishlist"""
        params = {
            "set_num": set_num,
            "priority": priority
        }
        if max_price:
            params["max_price"] = max_price
        
        response = self.session.post(f"{self.base_url}/users/{user_id}/wishlist", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict:
        """Get API performance metrics"""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()

def demo_workflow():
    """Demonstrate a complete user workflow"""
    client = LegoAPIClient()
    
    print("🧱 LEGO Recommendation API Demo")
    print("=" * 40)
    
    # 1. Health check
    print("\n1. Checking API health...")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
    except Exception as e:
        print(f"❌ API not available: {e}")
        return
    
    # 2. Get available themes
    print("\n2. Getting available themes...")
    try:
        themes = client.get_themes()
        print(f"📂 Found {len(themes)} themes")
        # Show first 5 themes
        for theme in themes[:5]:
            print(f"   - {theme['name']} (ID: {theme['id']}, {theme['set_count']} sets)")
    except Exception as e:
        print(f"❌ Error getting themes: {e}")
    
    # 3. Search for Star Wars sets
    print("\n3. Searching for Star Wars sets...")
    try:
        # First try searching by name
        star_wars_sets = client.search_sets(query="star wars", min_pieces=1, limit=5)
        
        # If no results, try searching by theme ID (158 is the main Star Wars theme)
        if not star_wars_sets:
            print("   Trying search by Star Wars theme ID...")
            star_wars_sets = client.search_sets(theme_ids=[158], limit=5)
        
        # If still no results, try other Star Wars theme IDs
        if not star_wars_sets:
            print("   Trying search by other Star Wars theme IDs...")
            star_wars_sets = client.search_sets(theme_ids=[18, 158, 209, 261], limit=5)
        
        print(f"🔍 Found {len(star_wars_sets)} Star Wars sets:")
        for set_info in star_wars_sets:
            print(f"   - {set_info['name']} ({set_info['set_num']}) - {set_info['num_parts']} pieces")
            
    except Exception as e:
        print(f"❌ Error searching sets: {e}")
        star_wars_sets = []
        
        # Fallback: search for any sets with decent piece count
        print("   Trying fallback search for any building sets...")
        try:
            star_wars_sets = client.search_sets(min_pieces=50, max_pieces=500, limit=5)
            print(f"🔍 Found {len(star_wars_sets)} building sets as fallback:")
            for set_info in star_wars_sets:
                print(f"   - {set_info['name']} ({set_info['set_num']}) - {set_info['num_parts']} pieces")
        except Exception as e2:
            print(f"❌ Fallback search also failed: {e2}")
    
    # 4. Create a demo user (if sets are available)
    recommendations = []
    if star_wars_sets:
        print("\n4. Creating demo user...")
        try:
            user_result = client.create_user(
                username=f"demo_user_{int(time.time())}",
                email=f"demo_{int(time.time())}@example.com",
                password="demo123"
            )
            user_id = user_result["user_id"]
            print(f"👤 Created user with ID: {user_id}")
            
            # 5. Rate some sets
            print("\n5. Rating some sets...")
            sample_set = star_wars_sets[0]
            try:
                client.track_interaction(
                    user_id=user_id,
                    set_num=sample_set["set_num"],
                    interaction_type="rating",
                    rating=5
                )
                print(f"⭐ Rated {sample_set['name']} with 5 stars")
            except Exception as e:
                print(f"❌ Error rating set: {e}")
            
            # 6. Get recommendations
            print("\n6. Getting personalized recommendations...")
            try:
                # Try hybrid first
                recommendations = client.get_recommendations(user_id=user_id, top_k=5)
                
                # If hybrid fails, try content-based with a set the user rated
                if not recommendations and star_wars_sets:
                    print("   Hybrid recommendations empty, trying content-based...")
                    recommendations = client.get_recommendations(
                        set_num=sample_set["set_num"], 
                        top_k=5, 
                        recommendation_type="content"
                    )
                
                # If still no results, try collaborative
                if not recommendations:
                    print("   Trying collaborative filtering...")
                    recommendations = client.get_recommendations(
                        user_id=user_id, 
                        top_k=5, 
                        recommendation_type="collaborative"
                    )
                
                print(f"🎯 Got {len(recommendations)} recommendations:")
                for rec in recommendations:
                    print(f"   - {rec['name']} (Score: {rec['score']:.2f})")
                    if rec['reasons']:
                        print(f"     Reasons: {', '.join(rec['reasons'])}")
                        
            except Exception as e:
                print(f"❌ Error getting recommendations: {e}")
                recommendations = []
            
            # 7. Add to wishlist
            if recommendations:
                print("\n7. Adding recommendation to wishlist...")
                try:
                    rec_set = recommendations[0]
                    client.add_to_wishlist(user_id=user_id, set_num=rec_set["set_num"], priority=1)
                    print(f"💝 Added {rec_set['name']} to wishlist")
                except Exception as e:
                    print(f"❌ Error adding to wishlist: {e}")
        
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            recommendations = []
    
    # 8. Get API metrics
    print("\n8. Getting API metrics...")
    try:
        metrics = client.get_metrics()
        print("📊 API Metrics:")
        print(f"   - Total requests: {metrics['total_requests']}")
        print(f"   - Avg response time: {metrics['avg_response_time']:.3f}s")
        print(f"   - Recommendations served: {metrics['total_recommendations_served']}")
        print(f"   - Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    import time
    demo_workflow()
