#!/usr/bin/env python3
"""
Test script for Brickbrain Gradio Interface

This script tests the connection to the Brickbrain API and validates
that all endpoints work correctly before launching the Gradio interface.
"""

import requests
import json
import sys
import time

API_BASE = "http://localhost:8000"

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {data.get('status')}")
            print(f"   Database: {data.get('database_status')}")
            print(f"   NLP: {data.get('nlp_status')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_natural_language_search():
    """Test natural language search endpoint"""
    print("\n🔍 Testing natural language search...")
    try:
        payload = {
            "query": "Star Wars sets for kids",
            "top_k": 3,
            "include_explanation": True
        }
        response = requests.post(f"{API_BASE}/search/natural", json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"✅ Natural language search works ({len(results)} results)")
            if results:
                print(f"   Sample result: {results[0]['name']} ({results[0]['set_num']})")
            return True
        else:
            print(f"❌ Natural language search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Natural language search error: {e}")
        return False

def test_query_understanding():
    """Test query understanding endpoint"""
    print("\n🧠 Testing query understanding...")
    try:
        payload = {"query": "birthday gift for 8 year old"}
        response = requests.post(f"{API_BASE}/nlp/understand", json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query understanding works")
            print(f"   Intent: {data.get('intent')}")
            print(f"   Confidence: {data.get('confidence', 0):.2f}")
            return True
        else:
            print(f"❌ Query understanding failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Query understanding error: {e}")
        return False

def test_conversational_ai():
    """Test conversational recommendations"""
    print("\n💬 Testing conversational AI...")
    try:
        payload = {
            "query": "I'm looking for a challenging build",
            "conversation_history": [],
            "context": {}
        }
        response = requests.post(f"{API_BASE}/recommendations/conversational", json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Conversational AI works")
            print(f"   Response type: {data.get('type')}")
            return True
        else:
            print(f"❌ Conversational AI failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Conversational AI error: {e}")
        return False

def test_semantic_similarity():
    """Test semantic similarity search"""
    print("\n🔗 Testing semantic similarity...")
    try:
        # Test using natural language search for text-based similarity
        payload = {
            "query": "sets similar to large detailed castle",
            "top_k": 3,
            "include_explanation": False
        }
        response = requests.post(f"{API_BASE}/search/natural", json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"✅ Semantic similarity works ({len(results)} results)")
            if results:
                print(f"   Sample result: {results[0]['name']} ({results[0]['set_num']})")
            return True
        else:
            print(f"❌ Semantic similarity failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Semantic similarity error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧱 Brickbrain API Test Suite")
    print("=" * 40)
    
    tests = [
        test_api_health,
        test_natural_language_search,
        test_query_understanding,
        test_conversational_ai,
        test_semantic_similarity
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Gradio interface should work perfectly.")
        print("🚀 You can now launch the Gradio interface:")
        print("   python3 gradio_launcher.py")
        print("   or")
        print("   ./launch_gradio.sh")
        return True
    else:
        print("⚠️  Some tests failed. Check your Docker Compose services:")
        print("   docker-compose ps")
        print("   docker-compose logs app")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
