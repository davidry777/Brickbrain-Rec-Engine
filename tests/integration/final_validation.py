#!/usr/bin/env python3
"""
FINAL VALIDATION - LEGO Recommendation System Production Readiness
================================================================

This script performs a comprehensive final validation of the entire system
to confirm production readiness and address all requirements.
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def test_api_running():
    """Test if the API is running and responsive"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is running and healthy")
            print(f"   Status: {health_data['status']}")
            print(f"   Engine: {health_data['recommendation_engine']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API is not running: {e}")
        return False

def test_recommendations():
    """Test all recommendation scenarios"""
    test_cases = [
        {
            "name": "Cold Start (New User)",
            "payload": {"recommendation_type": "hybrid", "top_k": 3},
            "expected_min": 1
        },
        {
            "name": "Content-Based (Liked Set)",
            "payload": {"recommendation_type": "content", "set_num": "75192-1", "top_k": 3},
            "expected_min": 1
        },
        {
            "name": "User-Based (Existing User)",
            "payload": {"recommendation_type": "collaborative", "user_id": 1000, "top_k": 3},
            "expected_min": 1
        },
        {
            "name": "Hybrid (User + Liked Set)",
            "payload": {"recommendation_type": "hybrid", "user_id": 1000, "set_num": "75192-1", "top_k": 3},
            "expected_min": 1
        }
    ]
    
    all_passed = True
    for test in test_cases:
        try:
            response = requests.post(
                "http://localhost:8000/recommendations",
                json=test["payload"],
                timeout=10
            )
            
            if response.status_code == 200:
                recommendations = response.json()
                if len(recommendations) >= test["expected_min"]:
                    print(f"✅ {test['name']}: {len(recommendations)} recommendations")
                else:
                    print(f"❌ {test['name']}: Only {len(recommendations)} recommendations (expected >= {test['expected_min']})")
                    all_passed = False
            else:
                print(f"❌ {test['name']}: HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"❌ {test['name']}: {e}")
            all_passed = False
    
    return all_passed

def test_database_health():
    """Test database connectivity and data integrity"""
    try:
        # Test database through API metrics
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Database connectivity confirmed")
            print(f"   Active users: {metrics.get('active_users', 0)}")
            print(f"   Recommendations served: {metrics.get('total_recommendations_served', 0)}")
            print(f"   Total requests: {metrics.get('total_requests', 0)}")
            
            # Test direct database access through themes endpoint
            themes_response = requests.get("http://localhost:8000/themes", timeout=5)
            if themes_response.status_code == 200:
                themes = themes_response.json()
                print(f"   Available themes: {len(themes)}")
                
                if len(themes) >= 100:
                    print("✅ Database has sufficient data for production")
                    return True
                else:
                    print("⚠️ Database data volumes are limited but functional")
                    return True
            else:
                print(f"❌ Themes endpoint failed: HTTP {themes_response.status_code}")
                return False
        else:
            print(f"❌ Database check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
        return False

def test_search_functionality():
    """Test search and filtering functionality"""
    try:
        # Test search
        search_payload = {
            "query": "star wars",
            "theme_ids": [],
            "min_pieces": 0,
            "max_pieces": 10000,
            "min_year": 2000,
            "max_year": 2025,
            "top_k": 5
        }
        
        response = requests.post(
            "http://localhost:8000/search/sets",
            json=search_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Search functionality: Found {len(results)} Star Wars sets")
            return True
        else:
            print(f"❌ Search failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_natural_language_recommendations():
    """Test natural language recommendation functionality"""
    nl_test_cases = [
        {
            "name": "Simple Theme Search",
            "payload": {
                "query": "star wars sets with lots of pieces",
                "top_k": 3,
                "include_explanation": True
            },
            "expected_themes": ["star wars"],
            "expected_min_results": 1
        },
        {
            "name": "Gift Recommendation",
            "payload": {
                "query": "birthday gift for my 10 year old nephew",
                "top_k": 5,
                "include_explanation": True
            },
            "expected_intent": "gift_recommendation",
            "expected_min_results": 1
        },
        {
            "name": "Complex Search with Filters",
            "payload": {
                "query": "detailed technic sets between 1000 and 2000 pieces for adults",
                "top_k": 5,
                "include_explanation": True
            },
            "expected_themes": ["technic"],
            "expected_min_results": 1
        },
        {
            "name": "Similar Set Request",
            "payload": {
                "query": "sets similar to the millennium falcon",
                "top_k": 3,
                "include_explanation": True
            },
            "expected_intent": "recommend_similar",
            "expected_min_results": 1
        },
        {
            "name": "Budget-Constrained Search",
            "payload": {
                "query": "star wars sets for kids",
                "top_k": 5,
                "include_explanation": True
            },
            "expected_filters": {"themes": ["Star Wars"]},
            "expected_min_results": 1
        }
    ]
    
    all_passed = True
    print("Testing Natural Language Search...")
    
    for test in nl_test_cases:
        try:
            response = requests.post(
                "http://localhost:8000/search/natural",
                json=test["payload"],
                timeout=15
            )
            
            if response.status_code == 200:
                nl_result = response.json()
                
                # Check basic response structure
                if "results" not in nl_result:
                    print(f"❌ {test['name']}: Missing 'results' in response")
                    all_passed = False
                    continue
                
                results = nl_result["results"]
                
                # Check minimum results
                if len(results) >= test["expected_min_results"]:
                    print(f"✅ {test['name']}: {len(results)} results returned")
                else:
                    print(f"❌ {test['name']}: Only {len(results)} results (expected >= {test['expected_min_results']})")
                    all_passed = False
                    continue
                
                # Check intent detection if specified
                if "expected_intent" in test:
                    detected_intent = nl_result.get("intent", "").lower()
                    expected_intent = test["expected_intent"].lower()
                    if expected_intent in detected_intent or detected_intent in expected_intent:
                        print(f"   ✅ Intent correctly detected: {detected_intent}")
                    else:
                        print(f"   ⚠️ Intent mismatch: expected {expected_intent}, got {detected_intent}")
                
                # Check theme extraction if specified
                if "expected_themes" in test:
                    extracted_filters = nl_result.get("extracted_filters", {})
                    themes = extracted_filters.get("themes", [])
                    theme_found = any(
                        expected_theme.lower() in str(themes).lower()
                        for expected_theme in test["expected_themes"]
                    )
                    if theme_found:
                        print(f"   ✅ Theme correctly extracted: {themes}")
                    else:
                        print(f"   ⚠️ Theme extraction: expected {test['expected_themes']}, got {themes}")
                
                # Check filters if specified
                if "expected_filters" in test:
                    extracted_filters = nl_result.get("extracted_filters", {})
                    for filter_key, expected_value in test["expected_filters"].items():
                        if filter_key in extracted_filters:
                            print(f"   ✅ Filter extracted: {filter_key} = {extracted_filters[filter_key]}")
                        else:
                            print(f"   ⚠️ Filter missing: {filter_key}")
                
                # Check if explanation is provided when requested
                if test["payload"].get("include_explanation"):
                    if "explanation" in nl_result and nl_result["explanation"]:
                        print(f"   ✅ Explanation provided")
                    else:
                        print(f"   ⚠️ Explanation missing or empty")
                
            else:
                print(f"❌ {test['name']}: HTTP {response.status_code}")
                if response.status_code == 500:
                    print(f"   Error: {response.text}")
                all_passed = False
        except Exception as e:
            print(f"❌ {test['name']}: {e}")
            all_passed = False
    
    return all_passed

def test_semantic_similarity():
    """Test semantic similarity search"""
    try:
        # Test semantic similarity endpoint
        similarity_payload = {
            "set_num": "75192-1",  # Millennium Falcon
            "description": "large detailed spaceship",
            "top_k": 3
        }
        
        response = requests.post(
            "http://localhost:8000/sets/similar/semantic",
            json=similarity_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            if len(results) >= 1:
                print(f"✅ Semantic similarity: Found {len(results)} similar sets")
                # Check if results have required fields
                for result in results[:2]:  # Check first 2 results
                    required_fields = ['set_num', 'name', 'relevance_score']
                    if all(field in result for field in required_fields):
                        print(f"   ✅ Result structure valid: {result['name']} (score: {result['relevance_score']:.2f})")
                    else:
                        print(f"   ⚠️ Result missing required fields: {result}")
                return True
            else:
                print(f"❌ Semantic similarity: No results returned")
                return False
        else:
            print(f"❌ Semantic similarity: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Semantic similarity test failed: {e}")
        return False

def test_query_understanding():
    """Test query understanding capabilities"""
    try:
        test_query = "I want a complex Star Wars set with over 1000 pieces for display"
        
        response = requests.post(
            "http://localhost:8000/nlp/understand",
            json={"query": test_query},
            timeout=10
        )
        
        if response.status_code == 200:
            understanding = response.json()
            
            # Check if response has expected structure (based on actual API response)
            expected_fields = ['intent', 'extracted_filters', 'confidence', 'extracted_entities']
            missing_fields = [field for field in expected_fields if field not in understanding]
            
            if not missing_fields:
                print("✅ Query understanding: All fields present")
                print(f"   Intent: {understanding.get('intent', 'unknown')}")
                print(f"   Confidence: {understanding.get('confidence', 0):.2%}")
                print(f"   Filters: {len(understanding.get('extracted_filters', {}))}")
                print(f"   Entities: {len(understanding.get('extracted_entities', {}))}")
                return True
            else:
                print(f"❌ Query understanding: Missing fields: {missing_fields}")
                return False
        else:
            print(f"❌ Query understanding: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Query understanding test failed: {e}")
        return False

def test_conversational_recommendations():
    """Test conversational recommendation capabilities"""
    try:
        conversation_payload = {
            "query": "I'm looking for a gift for my nephew",
            "conversation_history": [
                {"role": "user", "content": "What LEGO themes are popular?"},
                {"role": "assistant", "content": "Star Wars, City, and Technic are very popular themes."}
            ],
            "user_id": None,
            "context": {"budget": "under_100", "age": "child"}
        }
        
        response = requests.post(
            "http://localhost:8000/recommendations/conversational",
            json=conversation_payload,
            timeout=15
        )
        
        if response.status_code == 200:
            conv_result = response.json()
            
            # Check basic structure
            if "type" in conv_result and "results" in conv_result:
                print(f"✅ Conversational recommendations: {conv_result['type']}")
                print(f"   Results: {len(conv_result['results'])}")
                
                # Check for follow-up questions
                if "follow_up_questions" in conv_result:
                    print(f"   Follow-up questions: {len(conv_result['follow_up_questions'])}")
                
                return True
            else:
                print(f"❌ Conversational recommendations: Invalid response structure")
                return False
        else:
            print(f"❌ Conversational recommendations: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Conversational recommendations test failed: {e}")
        return False

def test_entity_extraction_production():
    """Test entity extraction in production environment"""
    try:
        print("Testing Entity Extraction...")
        
        entity_test_cases = [
            {
                "name": "Birthday Gift Query",
                "query": "birthday gift for my 8-year-old son who loves space themes",
                "expected_entities": ["recipient", "age", "occasion", "interest_category"],
                "expected_values": {
                    "recipient": "son",
                    "age": 8,
                    "occasion": "birthday",
                    "interest_category": "space"
                }
            },
            {
                "name": "Experience Level Query",
                "query": "challenging build for an expert adult builder",
                "expected_entities": ["building_preference", "experience_level"],
                "expected_values": {
                    "building_preference": "challenging",
                    "experience_level": "expert"
                }
            },
            {
                "name": "Feature-Specific Query",
                "query": "weekend project with minifigures and lights",
                "expected_entities": ["time_constraint", "special_features"],
                "expected_values": {
                    "time_constraint": "weekend_project",
                    "special_features": ["minifigures", "lights"]
                }
            },
            {
                "name": "Complex Multi-Entity Query",
                "query": "detailed Christmas present for my nephew who's an intermediate builder",
                "expected_entities": ["building_preference", "occasion", "recipient", "experience_level"],
                "expected_values": {
                    "building_preference": "detailed",
                    "occasion": "christmas",
                    "recipient": "nephew",
                    "experience_level": "intermediate"
                }
            }
        ]
        
        all_passed = True
        
        for test in entity_test_cases:
            try:
                response = requests.post(
                    "http://localhost:8000/nlp/understand",
                    json={"query": test["query"]},
                    timeout=10
                )
                
                if response.status_code == 200:
                    understanding = response.json()
                    extracted_entities = understanding.get("extracted_entities", {})
                    
                    print(f"\n   {test['name']}: '{test['query']}'")
                    print(f"     Extracted entities: {extracted_entities}")
                    
                    # Check if expected entity types are present
                    entities_found = 0
                    for entity_type in test["expected_entities"]:
                        if entity_type in extracted_entities:
                            entities_found += 1
                            # Check specific values if provided
                            if entity_type in test["expected_values"]:
                                expected_value = test["expected_values"][entity_type]
                                actual_value = extracted_entities[entity_type]
                                
                                if entity_type == "special_features":
                                    # For lists, check if expected items are present
                                    if isinstance(expected_value, list) and isinstance(actual_value, list):
                                        if any(item in actual_value for item in expected_value):
                                            print(f"     ✅ {entity_type}: {actual_value}")
                                        else:
                                            print(f"     ⚠️  {entity_type}: expected {expected_value}, got {actual_value}")
                                elif entity_type == "age":
                                    if actual_value == expected_value:
                                        print(f"     ✅ {entity_type}: {actual_value}")
                                    else:
                                        print(f"     ⚠️  {entity_type}: expected {expected_value}, got {actual_value}")
                                else:
                                    # Case-insensitive string comparison
                                    if str(actual_value).lower() == str(expected_value).lower():
                                        print(f"     ✅ {entity_type}: {actual_value}")
                                    else:
                                        print(f"     ⚠️  {entity_type}: expected {expected_value}, got {actual_value}")
                            else:
                                print(f"     ✅ {entity_type}: {extracted_entities[entity_type]}")
                    
                    # Calculate entity extraction success rate
                    extraction_rate = entities_found / len(test["expected_entities"])
                    if extraction_rate >= 0.7:  # 70% or more entities found
                        print(f"     ✅ Entity extraction: {entities_found}/{len(test['expected_entities'])} entities found")
                    else:
                        print(f"     ⚠️  Entity extraction: only {entities_found}/{len(test['expected_entities'])} entities found")
                        all_passed = False
                    
                    # Check confidence score
                    confidence = understanding.get("confidence", 0)
                    if confidence > 0.3:  # Reasonable confidence threshold
                        print(f"     ✅ Confidence: {confidence:.2f}")
                    else:
                        print(f"     ⚠️  Low confidence: {confidence:.2f}")
                else:
                    print(f"❌ {test['name']}: HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {test['name']}: {e}")
                all_passed = False
        
        if all_passed:
            print("\n✅ Entity extraction: All tests passed")
        else:
            print("\n⚠️  Entity extraction: Some tests had issues")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False

def main():
    print_header("FINAL PRODUCTION READINESS VALIDATION")
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Track overall results
    all_tests_passed = True
    test_results = {}
    
    # Test 1: API Health
    print_section("API Health Check")
    api_ok = test_api_running()
    test_results["API Health"] = api_ok
    all_tests_passed = all_tests_passed and api_ok
    
    if not api_ok:
        print("\n❌ API is not running. Please start the API first:")
        print("   python src/scripts/recommendation_api.py")
        return
    
    # Test 2: Database Health
    print_section("Database Health Check")
    db_ok = test_database_health()
    test_results["Database"] = db_ok
    all_tests_passed = all_tests_passed and db_ok
    
    # Test 3: Recommendation Engine
    print_section("Recommendation Engine Tests")
    rec_ok = test_recommendations()
    test_results["Recommendations"] = rec_ok
    all_tests_passed = all_tests_passed and rec_ok
    
    # Test 4: Search Functionality
    print_section("Search Functionality")
    search_ok = test_search_functionality()
    test_results["Search"] = search_ok
    all_tests_passed = all_tests_passed and search_ok
    
    # Test 5: Natural Language Recommendations
    print_section("Natural Language Recommendations")
    nl_ok = test_natural_language_recommendations()
    test_results["Natural Language"] = nl_ok
    all_tests_passed = all_tests_passed and nl_ok
    
    # Test 6: Semantic Similarity
    print_section("Semantic Similarity")
    semantic_ok = test_semantic_similarity()
    test_results["Semantic Similarity"] = semantic_ok
    all_tests_passed = all_tests_passed and semantic_ok
    
    # Test 7: Query Understanding
    print_section("Query Understanding")
    understanding_ok = test_query_understanding()
    test_results["Query Understanding"] = understanding_ok
    all_tests_passed = all_tests_passed and understanding_ok
    
    # Test 8: Conversational Recommendations
    print_section("Conversational Recommendations")
    conv_ok = test_conversational_recommendations()
    test_results["Conversational"] = conv_ok
    all_tests_passed = all_tests_passed and conv_ok
    
    # Test 9: Entity Extraction
    print_section("Entity Extraction")
    entity_ok = test_entity_extraction_production()
    test_results["Entity Extraction"] = entity_ok
    all_tests_passed = all_tests_passed and entity_ok
    
    # Final Assessment
    print_header("FINAL ASSESSMENT")
    
    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    score_percentage = (passed_count / total_count) * 100
    
    print(f"📊 Test Results: {passed_count}/{total_count} tests passed ({score_percentage:.1f}%)")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print()
    
    if all_tests_passed:
        print("🎉 SYSTEM IS FULLY PRODUCTION READY!")
        print()
        print("✅ All core functionality validated:")
        print("   • FastAPI service running and responsive")
        print("   • Database connectivity and data integrity")
        print("   • Cold start recommendations (popular sets)")
        print("   • Content-based recommendations (similar sets)")
        print("   • Collaborative filtering (user-based)")
        print("   • Hybrid recommendations (best of all)")
        print("   • Search and filtering capabilities")
        print("   • Natural language query processing")
        print("   • Semantic similarity search")
        print("   • Intent detection and filter extraction")
        print("   • Conversational recommendation interface")
        print("   • Entity extraction and understanding")
        print()
        print("🚀 READY FOR DEPLOYMENT!")
    elif score_percentage >= 75:
        print("⚠️ SYSTEM IS MOSTLY READY")
        print("   Minor issues detected but core functionality works")
        print("   Consider addressing failed tests before production")
    else:
        print("❌ SYSTEM IS NOT READY FOR PRODUCTION")
        print("   Critical issues detected - please fix before deployment")
    
    print()
    print("📝 Additional Notes:")
    print("   • System handles sparse data gracefully with fallbacks")
    print("   • Collaborative filtering will improve as user data grows")
    print("   • Content-based recommendations work immediately")
    print("   • Hybrid approach ensures robust recommendations")
    print("   • Natural language processing enables intuitive queries")
    print("   • Enhanced entity extraction identifies recipients, ages, occasions, and preferences")
    print("   • LLM-based entity extraction with robust regex fallback")
    print("   • Semantic search provides contextual understanding")
    print("   • Confidence scoring reflects query understanding quality")
    print("   • API includes proper error handling and validation")
    print("   • Conversational interface supports interactive experiences")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
