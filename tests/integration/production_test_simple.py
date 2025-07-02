#!/usr/bin/env python3
"""
Simple Production Test for LEGO Recommendation System
This simplified version tests core functionality without complex data generation
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import os
from datetime import datetime

# Add the scripts directory to path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scripts'))
from recommendation_system import HybridRecommender

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "brickbrain"),
    "user": os.getenv("DB_USER", "brickbrain"),
    "password": os.getenv("DB_PASSWORD", "brickbrain_password")
}

def test_system():
    """Test the system with minimal setup"""
    print("🧪 Simple Production Test - LEGO Recommendation System")
    print("=" * 60)
    
    # Connect to database
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Test 1: Check if we can connect and get basic data
        print("1️⃣ Testing database connection...")
        cursor.execute("SELECT COUNT(*) as count FROM sets")
        sets_count = cursor.fetchone()['count']
        print(f"   ✅ Found {sets_count} LEGO sets in database")
        
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE id >= 1000")
        users_count = cursor.fetchone()['count']
        print(f"   ✅ Found {users_count} test users in database")
        
        # Test 2: Test recommendation system initialization
        print("\n2️⃣ Testing recommendation system...")
        recommender = HybridRecommender(conn)
        print("   ✅ HybridRecommender initialized successfully")
        
        # Test 3: Test cold start recommendations (no user, no preferences)
        print("\n3️⃣ Testing cold start recommendations...")
        try:
            recommendations = recommender.get_recommendations(user_id=None, liked_set=None, top_k=5)
            print(f"   ✅ Cold start: Got {len(recommendations)} recommendations")
            if recommendations:
                print(f"      Example: {recommendations[0].set_num} - {recommendations[0].name}")
        except Exception as e:
            print(f"   ❌ Cold start failed: {e}")
        
        # Test 4: Test with liked set (content-based)
        print("\n4️⃣ Testing content-based recommendations...")
        try:
            recommendations = recommender.get_recommendations(user_id=None, liked_set="75192-1", top_k=5)
            print(f"   ✅ Content-based: Got {len(recommendations)} recommendations")
            if recommendations:
                print(f"      Example: {recommendations[0].set_num} - {recommendations[0].name}")
        except Exception as e:
            print(f"   ❌ Content-based failed: {e}")
        
        # Test 5: Test with existing user (if any)
        print("\n5️⃣ Testing user-based recommendations...")
        cursor.execute("SELECT id FROM users WHERE id >= 1000 LIMIT 1")
        user_row = cursor.fetchone()
        if user_row:
            try:
                recommendations = recommender.get_recommendations(user_id=user_row['id'], top_k=5)
                print(f"   ✅ User-based: Got {len(recommendations)} recommendations for user {user_row['id']}")
                if recommendations:
                    print(f"      Example: {recommendations[0].set_num} - {recommendations[0].name}")
            except Exception as e:
                print(f"   ❌ User-based failed: {e}")
        else:
            print("   ⚠️ No test users found, skipping user-based test")
        
        # Test 6: Check system metrics
        print("\n6️⃣ System metrics...")
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT u.id) as total_users,
                COUNT(DISTINCT ui.user_id) as active_users,
                COUNT(ui.id) as total_interactions,
                COUNT(DISTINCT ui.set_num) as rated_sets,
                ROUND(AVG(ui.rating), 2) as avg_rating
            FROM users u
            LEFT JOIN user_interactions ui ON u.id = ui.user_id
            WHERE u.id >= 1000
        """)
        metrics = cursor.fetchone()
        print(f"   📊 Total Users: {metrics['total_users']}")
        print(f"   📊 Active Users: {metrics['active_users']}")
        print(f"   📊 Total Interactions: {metrics['total_interactions']}")
        print(f"   📊 Rated Sets: {metrics['rated_sets']}")
        print(f"   📊 Average Rating: {metrics['avg_rating']}")
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("📋 OVERALL ASSESSMENT")
        print("=" * 60)
        
        readiness_score = 0
        max_score = 5
        
        # Check database connectivity
        if sets_count > 0:
            print("✅ Database connectivity: PASS")
            readiness_score += 1
        else:
            print("❌ Database connectivity: FAIL")
        
        # Check recommendation system
        try:
            test_recs = recommender.get_recommendations(user_id=None, liked_set=None, top_k=1)
            if test_recs:
                print("✅ Recommendation system: PASS")
                readiness_score += 1
            else:
                print("⚠️ Recommendation system: LIMITED (no recommendations returned)")
                readiness_score += 0.5
        except:
            print("❌ Recommendation system: FAIL")
        
        # Check content-based recommendations
        try:
            test_recs = recommender.get_recommendations(user_id=None, liked_set="75192-1", top_k=1)
            if test_recs:
                print("✅ Content-based recommendations: PASS")
                readiness_score += 1
            else:
                print("⚠️ Content-based recommendations: LIMITED")
                readiness_score += 0.5
        except:
            print("❌ Content-based recommendations: FAIL")
        
        # Check data volume
        if metrics['total_interactions'] and metrics['total_interactions'] > 100:
            print("✅ Data volume: SUFFICIENT")
            readiness_score += 1
        elif metrics['total_interactions'] and metrics['total_interactions'] > 10:
            print("⚠️ Data volume: LIMITED")
            readiness_score += 0.5
        else:
            print("❌ Data volume: INSUFFICIENT")
        
        # Check user base
        if metrics['active_users'] and metrics['active_users'] > 20:
            print("✅ User base: SUFFICIENT")
            readiness_score += 1
        elif metrics['active_users'] and metrics['active_users'] > 5:
            print("⚠️ User base: LIMITED")
            readiness_score += 0.5
        else:
            print("❌ User base: INSUFFICIENT")
        
        percentage = (readiness_score / max_score) * 100
        print(f"\n🎯 READINESS SCORE: {readiness_score}/{max_score} ({percentage:.1f}%)")
        
        if percentage >= 80:
            print("🎉 SYSTEM IS PRODUCTION READY!")
        elif percentage >= 60:
            print("⚠️ SYSTEM IS PARTIALLY READY - Some limitations exist")
        else:
            print("❌ SYSTEM IS NOT READY FOR PRODUCTION")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    test_system()
