#!/usr/bin/env python3
import os
import sys
import psycopg2
import logging
import sqlite3
import unittest
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.scripts.lego_nlp_recommeder import NLPRecommender, ConversationContext

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_db():
    """Connect to the PostgreSQL database"""
    try:
        # Get database connection parameters from environment variables
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            database=os.environ.get("DB_NAME", "brickbrain"),
            user=os.environ.get("DB_USER", "brickbrain"),
            password=os.environ.get("DB_PASSWORD", "brickbrain_password"),
            port=os.environ.get("DB_PORT", "5432")
        )
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        sys.exit(1)

def create_test_database():
    """Create a test SQLite database for unit testing"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE sets (
            set_num TEXT PRIMARY KEY,
            name TEXT,
            year INTEGER,
            num_parts INTEGER,
            theme_id INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE themes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            parent_id INTEGER
        )
    ''')
    
    # Insert test data
    cursor.execute("INSERT INTO themes VALUES (1, 'Star Wars', NULL)")
    cursor.execute("INSERT INTO themes VALUES (2, 'City', NULL)")
    cursor.execute("INSERT INTO sets VALUES ('75309-1', 'Republic Gunship', 2021, 3292, 1)")
    cursor.execute("INSERT INTO sets VALUES ('60321-1', 'Fire Rescue Helicopter', 2022, 212, 2)")
    
    conn.commit()
    return conn

class TestConversationMemory(unittest.TestCase):
    """Unit tests for conversation memory functionality"""
    
    def setUp(self):
        """Set up test database and recommender."""
        self.db_conn = create_test_database()
        self.recommender = NLPRecommender(self.db_conn, use_openai=False)
    
    def tearDown(self):
        """Clean up test database."""
        self.db_conn.close()
    
    def test_conversation_memory_initialization(self):
        """Test that conversation memory is properly initialized."""
        self.assertIsNotNone(self.recommender.conversation_memory)
        self.assertIsInstance(self.recommender.user_context, dict)
        self.assertIn('preferences', self.recommender.user_context)
        self.assertIn('previous_searches', self.recommender.user_context)
        self.assertIn('conversation_session', self.recommender.user_context)
        print("✅ Conversation memory initialization test passed")
    
    def test_add_to_conversation_memory(self):
        """Test adding interactions to conversation memory."""
        user_input = "I'm looking for Star Wars sets"
        ai_response = "Found 5 Star Wars sets"
        
        # Add to memory
        self.recommender.add_to_conversation_memory(user_input, ai_response)
        
        # Check memory was updated
        self.assertEqual(len(self.recommender.user_context['previous_searches']), 1)
        
        search_entry = self.recommender.user_context['previous_searches'][0]
        self.assertEqual(search_entry['query'], user_input)
        self.assertIn('timestamp', search_entry)
        self.assertIn('response_summary', search_entry)
        print("✅ Add to conversation memory test passed")
    
    def test_conversation_context_retrieval(self):
        """Test getting conversation context."""
        # Add some interactions
        self.recommender.add_to_conversation_memory("First query", "First response")
        self.recommender.add_to_conversation_memory("Second query", "Second response")
        
        # Get context
        context = self.recommender.get_conversation_context()
        
        # Verify context structure
        self.assertIsInstance(context, ConversationContext)
        self.assertEqual(len(context.current_session_queries), 2)
        self.assertEqual(context.current_session_queries[0], "First query")
        self.assertEqual(context.current_session_queries[1], "Second query")
        print("✅ Conversation context retrieval test passed")
    
    def test_user_preference_updates(self):
        """Test updating user preferences."""
        preferences = {
            'themes': {'Star Wars': 1},
            'recipient': 'nephew',
            'age': 10
        }
        
        self.recommender.update_user_preferences(preferences)
        
        # Check preferences were updated
        self.assertEqual(self.recommender.user_context['preferences']['themes'], {'Star Wars': 1})
        self.assertEqual(self.recommender.user_context['preferences']['recipient'], 'nephew')
        self.assertEqual(self.recommender.user_context['preferences']['age'], 10)
        print("✅ User preference updates test passed")
    
    def test_user_feedback_recording(self):
        """Test recording user feedback."""
        set_num = '75309-1'
        feedback = 'liked'
        rating = 5
        
        self.recommender.record_user_feedback(set_num, feedback, rating)
        
        # Check feedback was recorded
        self.assertEqual(len(self.recommender.user_context['liked_sets']), 1)
        
        feedback_entry = self.recommender.user_context['liked_sets'][0]
        self.assertEqual(feedback_entry['set_num'], set_num)
        self.assertEqual(feedback_entry['feedback'], feedback)
        self.assertEqual(feedback_entry['rating'], rating)
        print("✅ User feedback recording test passed")
    
    def test_preference_learning_from_feedback(self):
        """Test that preferences are learned from user feedback."""
        # Record positive feedback for Star Wars set
        self.recommender.record_user_feedback('75309-1', 'liked', 5)
        
        # Check that Star Wars theme preference was increased
        theme_prefs = self.recommender.user_context['preferences'].get('themes', {})
        self.assertIn('Star Wars', theme_prefs)
        self.assertGreater(theme_prefs['Star Wars'], 0)
        print("✅ Preference learning from feedback test passed")
    
    def test_context_aware_query_processing(self):
        """Test context-aware query processing."""
        # Set up some context
        self.recommender.update_user_preferences({'themes': {'Star Wars': 2}})
        self.recommender.add_to_conversation_memory("Star Wars sets", "Found sets")
        
        # Process query with context
        query = "Show me another set"
        result = self.recommender.process_nl_query_with_context(query)
        
        # Check that context influenced the result
        self.assertIsNotNone(result)
        self.assertIn('intent', result.__dict__)
        self.assertIn('confidence', result.__dict__)
        
        # Context should boost confidence
        self.assertGreater(result.confidence, 0.1)
        print("✅ Context-aware query processing test passed")
    
    def test_conversation_memory_clearing(self):
        """Test clearing conversation memory."""
        # Add some data
        self.recommender.add_to_conversation_memory("Test query", "Test response")
        self.recommender.update_user_preferences({'themes': {'Star Wars': 1}})
        
        # Clear memory
        self.recommender.clear_conversation_memory()
        
        # Check memory was cleared
        self.assertEqual(len(self.recommender.user_context['previous_searches']), 0)
        self.assertEqual(len(self.recommender.user_context['preferences']), 0)
        
        # Check conversation memory was cleared
        memory_vars = self.recommender.conversation_memory.load_memory_variables({})
        self.assertEqual(len(memory_vars.get('chat_history', [])), 0)
        print("✅ Conversation memory clearing test passed")
    
    def test_confidence_boost_with_context(self):
        """Test that context boosts confidence scores."""
        # Query without context
        query = "LEGO sets"
        result_no_context = self.recommender.process_nl_query(query, None)
        
        # Add context
        self.recommender.update_user_preferences({'themes': {'Star Wars': 1}})
        self.recommender.add_to_conversation_memory("Previous query", "Previous response")
        
        # Query with context
        result_with_context = self.recommender.process_nl_query_with_context(query)
        
        # Context should boost confidence
        self.assertGreater(result_with_context.confidence, result_no_context.confidence)
        print("✅ Confidence boost with context test passed")

def test_conversation_memory_suite():
    """Run all conversation memory tests"""
    print("\n" + "="*60)
    print("🧠 TESTING CONVERSATION MEMORY")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConversationMemory)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All conversation memory tests passed!")
        return True
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")
        return False

def test_entity_extraction():
    """Test entity extraction functionality"""
    print("\n" + "="*60)
    print("🏷️  TESTING ENTITY EXTRACTION")
    print("="*60)
    
    # Create a mock database connection for entity testing
    class MockDBConnection:
        def __init__(self):
            pass
        def cursor(self):
            return None
    
    # Initialize NLP recommender
    use_openai = os.environ.get("USE_OPENAI", "false").lower() == "true"
    recommender = NLPRecommender(MockDBConnection(), use_openai=use_openai)
    
    # Test cases for entity extraction
    entity_test_cases = [
        {
            "query": "Birthday gift for my 8-year-old son who loves space themes",
            "expected_entities": {
                "recipient": "son",
                "age": 8,
                "occasion": "birthday",
                "interest_category": "space"
            }
        },
        {
            "query": "Christmas present for my daughter, she's a beginner with LEGO",
            "expected_entities": {
                "recipient": "daughter",
                "occasion": "christmas",
                "experience_level": "beginner"
            }
        },
        {
            "query": "Challenging build for an expert adult builder with vehicles",
            "expected_entities": {
                "building_preference": "challenging",
                "experience_level": "expert",
                "interest_category": "vehicles"
            }
        },
        {
            "query": "Quick weekend project with minifigures for my nephew",
            "expected_entities": {
                "recipient": "nephew",
                "building_preference": "quick_build",
                "special_features": ["minifigures"],
                "time_constraint": "weekend_project"
            }
        },
        {
            "query": "Detailed motorized set for a 12-year-old who likes robots",
            "expected_entities": {
                "age": 12,
                "building_preference": "detailed",
                "special_features": ["motorized"],
                "interest_category": "robots"
            }
        }
    ]
    
    print("Testing Enhanced Entity Extraction:")
    print("-" * 40)
    
    all_tests_passed = True
    
    for i, test_case in enumerate(entity_test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected_entities"]
        
        print(f"\n{i}. Query: '{query}'")
        
        # Test regex-based extraction
        entities_regex = recommender._extract_entities_regex(query)
        print(f"   Regex entities: {entities_regex}")
        
        # Test LLM-based extraction (falls back to regex if LLM unavailable)
        entities_llm = recommender._extract_entities_llm(query)
        print(f"   LLM entities: {entities_llm}")
        
        # Test intent detection
        intent = recommender._detect_intent(query)
        print(f"   Intent: {intent}")
        
        # Test semantic query creation
        semantic_query = recommender._create_semantic_query(query, {}, entities_llm)
        print(f"   Semantic query: {semantic_query[:80]}...")
        
        # Validate key entities are extracted
        entities_found = entities_llm or entities_regex
        test_passed = True
        
        for expected_key, expected_value in expected.items():
            if expected_key in entities_found:
                actual_value = entities_found[expected_key]
                if expected_key == "special_features":
                    # For lists, check if expected items are present
                    if isinstance(expected_value, list) and isinstance(actual_value, list):
                        if not any(item in actual_value for item in expected_value):
                            test_passed = False
                            print(f"   ❌ Missing expected {expected_key}: {expected_value}")
                    else:
                        test_passed = False
                elif actual_value != expected_value:
                    # For exact matches, allow some flexibility
                    if expected_key == "age" and isinstance(actual_value, int) and actual_value == expected_value:
                        continue
                    elif expected_key in ["recipient", "occasion", "experience_level", "building_preference", "interest_category"]:
                        if str(actual_value).lower() == str(expected_value).lower():
                            continue
                    test_passed = False
                    print(f"   ❌ {expected_key}: expected '{expected_value}', got '{actual_value}'")
            else:
                print(f"   ⚠️  Missing entity: {expected_key}")
        
        if test_passed:
            print(f"   ✅ Entity extraction test passed")
        else:
            all_tests_passed = False
        
        print("-" * 30)
    
    return all_tests_passed

def test_nlp_recommender():
    """Test the NLPRecommender with various queries"""
    print("\n" + "="*60)
    print("🧠 TESTING NLP RECOMMENDER")
    print("="*60)
    
    # Connect to database
    conn = connect_to_db()
    
    # Initialize NLPRecommender
    # Set use_openai=True if you have OpenAI API key and want to use it
    use_openai = os.environ.get("USE_OPENAI", "false").lower() == "true"
    recommender = NLPRecommender(conn, use_openai=use_openai)
    
    # Prepare vector database (limit to 500 sets for faster testing)
    recommender.prep_vectorDB(limit_sets=500)
    
    # Test queries with enhanced entity extraction focus
    test_queries = [
        "I need a Star Wars set with around 500 pieces",
        "What's a good birthday gift for a 10-year old who likes Technic?",
        "Show me complex sets from the last few years for expert builders",
        "I'm looking for simple City sets under $50 for my daughter",
        "Christmas present for my nephew, detailed motorized vehicles",
        "Weekend project with minifigures for a beginner"
    ]
    
    # Process each query
    for query in test_queries:
        logger.info(f"\n\nProcessing query: '{query}'")
        
        # Process the NL query to extract intent, filters, etc.
        nl_result = recommender.process_nl_query(query, user_context=None)
        
        logger.info(f"Detected intent: {nl_result.intent}")
        logger.info(f"Extracted filters: {nl_result.filters}")
        logger.info(f"Extracted entities: {nl_result.extracted_entities}")
        logger.info(f"Confidence score: {nl_result.confidence}")
        logger.info(f"Semantic query: '{nl_result.semantic_query}'")
        
        # Get recommendations
        results = recommender.semantic_search(query, top_k=5)
        
        # Display results
        logger.info(f"Top recommendations for '{query}':")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. {result['name']} ({result['set_num']}) - {result['num_parts']} pieces - {result['theme']} ({result['year']})")
        
        print("\n" + "-"*80)
    
    conn.close()

def run_all_tests():
    """Run all NLP recommender tests"""
    print("="*80)
    print("🧪 NLP RECOMMENDER COMPREHENSIVE TESTS")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Conversation Memory
    try:
        memory_test_passed = test_conversation_memory_suite()
        if memory_test_passed:
            print("\n✅ Conversation memory tests PASSED")
        else:
            print("\n⚠️  Some conversation memory tests had issues")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Conversation memory tests FAILED: {e}")
        all_passed = False
    
    # Test 2: Entity Extraction
    try:
        entity_test_passed = test_entity_extraction()
        if entity_test_passed:
            print("\n✅ Entity extraction tests PASSED")
        else:
            print("\n⚠️  Some entity extraction tests had issues")
            all_passed = False
    except Exception as e:
        print(f"\n❌ Entity extraction tests FAILED: {e}")
        all_passed = False
    
    # Test 3: Full NLP Recommender (if database available)
    try:
        test_nlp_recommender()
        print("\n✅ NLP recommender tests COMPLETED")
    except Exception as e:
        print(f"\n❌ NLP recommender tests FAILED: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests had issues - check output above")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
