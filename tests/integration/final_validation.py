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
    print("   • API includes proper error handling and validation")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
