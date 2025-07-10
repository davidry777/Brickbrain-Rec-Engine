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

def test_nlp_recommender():
    """Test the NLPRecommender with various queries"""
    # Connect to database
    conn = connect_to_db()
    
    # Initialize NLPRecommender
    # Set use_openai=True if you have OpenAI API key and want to use it
    use_openai = os.environ.get("USE_OPENAI", "false").lower() == "true"
    recommender = NLPRecommender(conn, use_openai=use_openai)
    
    # Prepare vector database (limit to 500 sets for faster testing)
    recommender.prep_vectorDB(limit_sets=500)
    
    # Test queries
    test_queries = [
        "I need a Star Wars set with around 500 pieces",
        "What's a good gift for a 10-year old who likes Technic?",
        "Show me complex sets from the last few years",
        "I'm looking for City sets under $50"
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

if __name__ == "__main__":
    test_nlp_recommender()
