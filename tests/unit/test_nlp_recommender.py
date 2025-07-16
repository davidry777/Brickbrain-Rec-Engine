#!/usr/bin/env python3
import os
import sys
import psycopg2
import logging
from dotenv import load_dotenv
from src.scripts.lego_nlp_recommeder import NLPRecommender

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
    
    # Test 1: Entity Extraction
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
    
    # Test 2: Full NLP Recommender (if database available)
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
