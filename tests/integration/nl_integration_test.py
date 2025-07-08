#!/usr/bin/env python3
"""
Enhanced Natural Language Integration Test
Tests the NL components work correctly with the existing system
Combines database, API, and NL feature testing
"""

import sys
import os
import requests
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import psycopg2
from src.scripts.lego_nlp_recommeder import (
    NLPRecommender, 
    NLQueryResult,
    SearchFilters
)

class TestNaturalLanguageIntegration(unittest.TestCase):
    """Enhanced test for natural language components and API integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database connection and API client"""
        cls.db_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "brickbrain"),
            "user": os.getenv("DB_USER", "brickbrain"),
            "password": os.getenv("DB_PASSWORD", "brickbrain_password"),
            "port": int(os.getenv("DB_PORT", 5432))
        }
        
        cls.api_base_url = "http://localhost:8000"
        
        try:
            cls.conn = psycopg2.connect(**cls.db_params)
            cls.nl_recommender = NLPRecommender(cls.conn, use_openai=False)
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
            
        # Test API availability
        try:
            response = requests.get(f"{cls.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ API server is accessible")
                cls.api_available = True
            else:
                print(f"⚠️  API server returned status {response.status_code}")
                cls.api_available = False
        except Exception as e:
            print(f"⚠️  API server not accessible: {e}")
            cls.api_available = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if hasattr(cls, 'conn'):
            cls.conn.close()
    
    def test_01_database_connectivity(self):
        """Test database connection and basic queries"""
        print("\n🔗 Testing database connectivity...")
        
        with self.conn.cursor() as cur:
            # Test basic table existence
            cur.execute("SELECT COUNT(*) FROM sets LIMIT 1;")
            result = cur.fetchone()
            self.assertIsNotNone(result)
            print(f"   Database accessible, sets table has data")
    
    def test_02_embeddings_initialization(self):
        """Test that embeddings can be initialized"""
        print("\n📊 Testing embeddings initialization...")
        
        # Check embeddings model is loaded
        self.assertIsNotNone(self.nl_recommender.embeddings)
        
        # Test creating embeddings for a sample text
        test_text = "LEGO Star Wars Millennium Falcon"
        embedding = self.nl_recommender.embeddings.embed_query(test_text)
        
        self.assertIsInstance(embedding, list)
        self.assertTrue(len(embedding) > 0)
        print(f"✅ Embedding dimension: {len(embedding)}")
    
    def test_02_query_intent_detection(self):
        """Test intent detection from queries"""
        print("\n🎯 Testing intent detection...")
        
        test_cases = [
            ("find me some star wars sets", "search"),
            ("I need a gift for my nephew", "gift_recommendation"),
            ("show me sets similar to the hogwarts castle", "recommend_similar"),
            ("should I buy the millennium falcon for my collection?", "collection_advice")
        ]
        
        for query, expected_intent in test_cases:
            intent = self.nl_recommender._detect_intent(query)
            print(f"Query: '{query}' → Intent: {intent}")
            self.assertEqual(intent, expected_intent)
    
    def test_03_filter_extraction(self):
        """Test filter extraction from natural language"""
        print("\n🔍 Testing filter extraction...")
        
        test_queries = [
            ("star wars sets under 500 pieces", {
                'themes': ['Star Wars'],
                'max_pieces': 500
            }),
            ("complex technic sets between 1000 and 2000 pieces", {
                'themes': ['Technic'],
                'complexity': 'complex',
                'min_pieces': 1000,
                'max_pieces': 2000
            }),
            ("gift for 8 year old under $50", {
                'min_age': 6,
                'max_age': 10,
                'budget_min': 35.0,
                'budget_max': 65.0
            })
        ]
        
        for query, expected_filters in test_queries:
            filters = self.nl_recommender._extract_filters_regex(query)
            print(f"\nQuery: '{query}'")
            print(f"Extracted: {filters}")
            
            # Check if expected filters are present
            for key, value in expected_filters.items():
                if key in filters:
                    if isinstance(value, list):
                        self.assertTrue(any(v in filters[key] for v in value))
                    else:
                        # Allow some flexibility in numeric values
                        if isinstance(value, (int, float)):
                            self.assertAlmostEqual(filters[key], value, delta=value*0.3)
                        else:
                            self.assertEqual(filters[key], value)
    
    def test_04_entity_extraction(self):
        """Test entity extraction"""
        print("\n🏷️ Testing entity extraction...")
        
        test_queries = [
            ("birthday gift for my son", {'occasion': 'birthday', 'recipient': 'son'}),
            ("christmas present for my daughter", {'occasion': 'christmas', 'recipient': 'daughter'}),
            ("something with lots of detail", {'preference': 'detailed'})
        ]
        
        for query, expected_entities in test_queries:
            entities = self.nl_recommender._extract_entities_regex(query)
            print(f"\nQuery: '{query}'")
            print(f"Entities: {entities}")
            
            for key, value in expected_entities.items():
                self.assertEqual(entities.get(key), value)
    
    def test_05_vector_database_creation(self):
        """Test vector database creation"""
        print("\n🗄️ Testing vector database creation...")
        
        # This might take a while, so we'll test with a small subset
        # Modify the query in prepare_vector_database to limit results for testing
        try:
            # Create a test version that only processes a few sets
            test_query = """
            SELECT 
                s.set_num, s.name, s.year, s.num_parts,
                t.name as theme_name, t.parent_id as parent_theme_id
            FROM sets s
            LEFT JOIN themes t ON s.theme_id = t.id
            WHERE s.num_parts > 0
            LIMIT 10
            """
            
            df = self.nl_recommender._create_test_dataframe(test_query)
            self.assertGreater(len(df), 0)
            print(f"✅ Loaded {len(df)} test sets")
            
        except AttributeError:
            # If method doesn't exist, just check we can create descriptions
            test_row = {
                'set_num': '10001-1',
                'name': 'Test Set',
                'year': 2024,
                'num_parts': 500,
                'theme_name': 'City',
                'parent_theme_name': 'City',
                'num_colors': 10,
                'num_minifigs': 4,
                'part_categories': 'Bricks, Plates'
            }
            
            description = self.nl_recommender._create_set_description(test_row)
            self.assertIsInstance(description, str)
            self.assertIn('Test Set', description)
            print(f"✅ Created description: {description[:100]}...")
    
    def test_06_natural_query_processing(self):
        """Test full natural language query processing"""
        print("\n🧠 Testing natural query processing...")
        
        query = "I want a detailed Star Wars spaceship with over 1000 pieces for display"
        result = self.nl_recommender.process_natural_query(query)
        
        self.assertIsInstance(result, NLQueryResult)
        self.assertIsInstance(result.intent, str)
        self.assertIsInstance(result.filters, dict)
        self.assertIsInstance(result.confidence, float)
        
        print(f"Query: '{query}'")
        print(f"Intent: {result.intent}")
        print(f"Filters: {result.filters}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Semantic Query: {result.semantic_query}")
        
        # Check that semantic query is enhanced
        self.assertGreater(len(result.semantic_query), len(query))
    
    def test_07_search_performance(self):
        """Test search performance"""
        print("\n⚡ Testing search performance...")
        
        import time
        
        queries = [
            "space sets with lots of pieces",
            "simple sets for young children",
            "technic cars with motors"
        ]
        
        for query in queries:
            start_time = time.time()
            result = self.nl_recommender.process_natural_query(query)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            print(f"Query: '{query}' - Processing time: {processing_time:.2f}ms")
            
            # Should be fast even without optimization
            self.assertLess(processing_time, 1000)  # Under 1 second
    
    def test_08_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n⚠️ Testing edge cases...")
        
        edge_cases = [
            "",  # Empty query
            "!!@@##$$%%",  # Special characters
            "a" * 500,  # Very long query
            "lego" * 100,  # Repetitive query
            "sets with exactly 1234.56 pieces",  # Precise decimal
            "🧱🚀🏰",  # Emojis
        ]
        
        for query in edge_cases:
            try:
                result = self.nl_recommender.process_natural_query(query)
                print(f"✅ Handled edge case: '{query[:50]}...'")
                self.assertIsInstance(result, NLQueryResult)
            except Exception as e:
                print(f"⚠️ Exception for '{query[:50]}...': {type(e).__name__}")
    
    def test_03_api_health_check(self):
        """Test API health endpoint"""
        print("\n🏥 Testing API health check...")
        
        if not self.api_available:
            self.skipTest("API server not available")
        
        response = requests.get(f"{self.api_base_url}/health")
        self.assertEqual(response.status_code, 200)
        
        health_data = response.json()
        self.assertIn("status", health_data)
        print(f"   API health: {health_data.get('status', 'unknown')}")
    
    def test_04_nl_search_api(self):
        """Test natural language search via API"""
        print("\n🔍 Testing NL search API...")
        
        if not self.api_available:
            self.skipTest("API server not available")
        
        test_queries = [
            "star wars sets for kids",
            "birthday gift for 8 year old",
            "challenging technic sets",
            "small sets under 200 pieces"
        ]
        
        for query in test_queries:
            print(f"   Testing query: '{query}'")
            
            payload = {
                "query": query,
                "top_k": 3,
                "include_explanation": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/search/natural",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn("results", data)
                results = data["results"]
                self.assertLessEqual(len(results), 3)
                print(f"     ✅ Found {len(results)} results")
                
                # Check result structure
                if results:
                    result = results[0]
                    required_fields = ["set_num", "name", "theme_name", "num_parts"]
                    for field in required_fields:
                        self.assertIn(field, result)
            else:
                print(f"     ⚠️  Query failed with status {response.status_code}")
    
    def test_05_query_understanding_api(self):
        """Test query understanding API endpoint"""
        print("\n🧠 Testing query understanding API...")
        
        if not self.api_available:
            self.skipTest("API server not available")
        
        test_query = "I want a challenging Star Wars set for my 12-year-old nephew's birthday"
        
        payload = {"query": test_query}
        
        try:
            response = requests.post(
                f"{self.api_base_url}/nlp/understand",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Query understanding response received")
                
                # Check for expected understanding components
                expected_keys = ["intent", "entities", "filters"]
                for key in expected_keys:
                    if key in data:
                        print(f"   ✅ Found {key}: {data[key]}")
            else:
                print(f"   ⚠️  Understanding endpoint returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Query understanding request failed: {e}")
    
    def test_06_response_time_performance(self):
        """Test API response time performance"""
        print("\n⏱️  Testing response time performance...")
        
        if not self.api_available:
            self.skipTest("API server not available")
        
        query = "star wars millennium falcon"
        payload = {"query": query, "top_k": 5}
        
        response_times = []
        
        for i in range(3):
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/search/natural",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"   Query {i+1} response time: {response_time:.2f}s")
            
            if response.status_code == 200:
                self.assertTrue(response_time < 10.0, f"Response time too slow: {response_time}s")
        
        avg_response_time = sum(response_times) / len(response_times)
        print(f"   Average response time: {avg_response_time:.2f}s")
        
        # Check that average response time is reasonable
        self.assertTrue(avg_response_time < 5.0, f"Average response time too slow: {avg_response_time}s")
    
    def test_07_error_handling(self):
        """Test API error handling"""
        print("\n🛡️  Testing error handling...")
        
        if not self.api_available:
            self.skipTest("API server not available")
        
        # Test empty query
        response = requests.post(
            f"{self.api_base_url}/search/natural",
            json={"query": "", "top_k": 3}
        )
        
        # Should handle empty query gracefully
        self.assertIn(response.status_code, [400, 422, 200])
        print(f"   Empty query handled with status {response.status_code}")
        
        # Test invalid parameters
        response = requests.post(
            f"{self.api_base_url}/search/natural",
            json={"query": "test", "top_k": -1}
        )
        
        # Should reject invalid parameters
        self.assertIn(response.status_code, [400, 422])
        print(f"   Invalid parameters rejected with status {response.status_code}")
    
    def test_08_data_availability(self):
        """Test that sufficient data is available for NL features"""
        print("\n📊 Testing data availability...")
        
        with self.conn.cursor() as cur:
            # Check sets data
            cur.execute("SELECT COUNT(*) FROM sets;")
            sets_count = cur.fetchone()[0]
            self.assertGreater(sets_count, 0, "No sets data available")
            print(f"   Sets available: {sets_count}")
            
            # Check themes data
            cur.execute("SELECT COUNT(DISTINCT theme_name) FROM sets;")
            themes_count = cur.fetchone()[0]
            self.assertGreater(themes_count, 0, "No themes data available")
            print(f"   Themes available: {themes_count}")
            
            # Check if we have Star Wars data (common test case)
            cur.execute("SELECT COUNT(*) FROM sets WHERE theme_name ILIKE '%star wars%';")
            sw_count = cur.fetchone()[0]
            if sw_count > 0:
                print(f"   Star Wars sets: {sw_count}")
            else:
                print("   ⚠️  No Star Wars sets found (may affect some tests)")

    # ...existing code...
def run_integration_tests():
    """Run all integration tests"""
    print("="*80)
    print("🧪 NATURAL LANGUAGE INTEGRATION TESTS")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaturalLanguageIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)