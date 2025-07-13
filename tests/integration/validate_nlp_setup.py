#!/usr/bin/env python3
"""
Validation script for NLP setup in the LEGO Recommendation Engine.
This script verifies that all NLP components are properly configured and working.
"""

import os
import sys
import logging
import requests
import psycopg2
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            database=os.environ.get("DB_NAME", "brickbrain"),
            user=os.environ.get("DB_USER", "brickbrain"),
            password=os.environ.get("DB_PASSWORD", "brickbrain_password"),
            port=os.environ.get("DB_PORT", "5432")
        )
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sets LIMIT 1;")
        count = cur.fetchone()[0]
        conn.close()
        
        logger.info(f"✅ Database connected successfully. Found {count} LEGO sets.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_ollama_service() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if 'mistral:latest' in model_names:
                logger.info("✅ Ollama service is running with Mistral model available")
                return True
            else:
                logger.warning(f"⚠️ Ollama is running but Mistral model not found. Available models: {model_names}")
                return False
        else:
            logger.error(f"❌ Ollama service returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to Ollama service: {e}")
        return False

def check_nlp_packages() -> bool:
    """Check if required NLP packages are available"""
    required_packages = [
        "langchain",
        "langchain_huggingface",
        "langchain_ollama",
        "sentence_transformers",
        "faiss",
        "transformers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is missing")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        return False
    else:
        logger.info("✅ All required NLP packages are available")
        return True

def check_fastapi_server() -> bool:
    """Check if FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ FastAPI server is running and healthy")
            return True
        else:
            logger.error(f"❌ FastAPI server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to FastAPI server: {e}")
        return False

def test_nlp_recommender() -> bool:
    """Test the NLP recommender functionality"""
    try:
        # Import here to avoid import errors if packages are missing
        from src.scripts.lego_nlp_recommeder import NLPRecommender
        
        # Connect to database
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            database=os.environ.get("DB_NAME", "brickbrain"),
            user=os.environ.get("DB_USER", "brickbrain"),
            password=os.environ.get("DB_PASSWORD", "brickbrain_password"),
            port=os.environ.get("DB_PORT", "5432")
        )
        
        # Initialize recommender
        recommender = NLPRecommender(conn, use_openai=False)
        
        # Test query processing
        test_query = "I need a simple Star Wars set"
        nl_result = recommender.process_nl_query(test_query, None)
        
        logger.info(f"✅ NLP Query Processing successful:")
        logger.info(f"   Intent: {nl_result.intent}")
        logger.info(f"   Filters: {nl_result.filters}")
        logger.info(f"   Confidence: {nl_result.confidence}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ NLP Recommender test failed: {e}")
        return False

def main():
    """Run all validation checks"""
    logger.info("🧱 LEGO Recommendation Engine - NLP Setup Validation")
    logger.info("=" * 60)
    
    checks = [
        ("Database Connection", check_database_connection),
        ("NLP Packages", check_nlp_packages),
        ("Ollama Service", check_ollama_service),
        ("FastAPI Server", check_fastapi_server),
        ("NLP Recommender", test_nlp_recommender)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Running {check_name} check...")
        results[check_name] = check_func()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{check_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All checks passed! Your NLP setup is ready to use.")
        return 0
    else:
        logger.error("⚠️ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
