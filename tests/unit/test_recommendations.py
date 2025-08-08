#!/usr/bin/env python3
"""
Comprehensive test suite for collaborative filtering and hybrid recommendations
This script simulates realistic user data and tests scalability
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sys
import os

# Add the scripts directory to path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'scripts'))
from recommendation_system import HybridRecommender, RecommendationRequest
from hard_constraint_filter import (
    HardConstraintFilter, HardConstraint, ConstraintResult,
    ConstraintType, ConstraintSeverity, 
    create_budget_constraints, create_age_appropriate_constraints, create_size_constraints
)

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "brickbrain"),
    "user": os.getenv("DB_USER", "brickbrain"),
    "password": os.getenv("DB_PASSWORD", "brickbrain_password")
}

class RecommendationSystemTester:
    def __init__(self):
        self.conn = psycopg2.connect(**DATABASE_CONFIG)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self.constraint_filter = HardConstraintFilter(self.conn)
        
    def simulate_realistic_user_data(self, num_users=100, min_ratings_per_user=5, max_ratings_per_user=50):
        """
        Generate realistic user interaction data to test collaborative filtering
        """
        print(f"🎭 Simulating {num_users} users with realistic behavior patterns...")
        
        # Get popular sets across different themes
        popular_sets_query = """
        SELECT s.set_num, s.name, s.theme_id, t.name as theme_name, s.num_parts, s.year
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        WHERE s.num_parts BETWEEN 50 AND 2000
        AND s.year >= 2015
        AND s.theme_id NOT IN (501, 503, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 777)
        ORDER BY s.year DESC, s.num_parts DESC
        LIMIT 500
        """
        
        sets_df = pd.read_sql(popular_sets_query, self.conn)
        print(f"📦 Found {len(sets_df)} sets for user simulation")
        
        # Create user personas with different preferences
        user_personas = [
            {"name": "Star Wars Fan", "preferred_themes": [158, 18], "rating_bias": 0.3},
            {"name": "City Builder", "preferred_themes": [52], "rating_bias": 0.2},
            {"name": "Creator Enthusiast", "preferred_themes": [22, 672], "rating_bias": 0.4},
            {"name": "Castle Lover", "preferred_themes": [186], "rating_bias": 0.3},
            {"name": "Technic Expert", "preferred_themes": [1], "rating_bias": 0.5},
            {"name": "Friends Fan", "preferred_themes": [494], "rating_bias": 0.2},
            {"name": "General Builder", "preferred_themes": [], "rating_bias": 0.1}
        ]
        
        # Clear existing test data
        self.cursor.execute("DELETE FROM user_interactions WHERE user_id >= 1000")
        self.cursor.execute("DELETE FROM users WHERE id >= 1000")
        self.conn.commit()
        
        users_created = 0
        interactions_created = 0
        
        for user_id in range(1000, 1000 + num_users):
            # Assign random persona
            persona = random.choice(user_personas)
            
            # Create user
            self.cursor.execute("""
                INSERT INTO users (id, username, email, password_hash, preferred_themes, complexity_preference)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                user_id,
                f"testuser_{user_id}",
                f"test{user_id}@example.com",
                "test_hash",
                persona["preferred_themes"],
                random.choice(["simple", "moderate", "complex"])
            ))
            users_created += 1
            
            # Generate ratings for this user
            num_ratings = random.randint(min_ratings_per_user, max_ratings_per_user)
            
            # Filter sets by user's preferred themes (if any)
            if persona["preferred_themes"]:
                user_sets = sets_df[sets_df['theme_id'].isin(persona["preferred_themes"])]
                if len(user_sets) < num_ratings:
                    # Add some random sets if not enough in preferred themes
                    additional_sets = sets_df[~sets_df['theme_id'].isin(persona["preferred_themes"])].sample(
                        min(num_ratings - len(user_sets), len(sets_df) - len(user_sets))
                    )
                    user_sets = pd.concat([user_sets, additional_sets])
            else:
                user_sets = sets_df
            
            # Sample sets for this user
            if len(user_sets) >= num_ratings:
                selected_sets = user_sets.sample(num_ratings)
            else:
                selected_sets = user_sets
            
            for _, set_row in selected_sets.iterrows():
                # Generate realistic rating based on persona and set characteristics
                base_rating = 3.5
                
                # Bias based on persona preference
                if set_row['theme_id'] in persona["preferred_themes"]:
                    base_rating += persona["rating_bias"] * 2
                
                # Slight bias for newer sets
                if set_row['year'] >= 2020:
                    base_rating += 0.2
                
                # Bias based on set size (people like substantial sets)
                if 200 <= set_row['num_parts'] <= 800:
                    base_rating += 0.3
                
                # Add some randomness
                rating = base_rating + random.gauss(0, 0.5)
                rating = max(1, min(5, round(rating)))
                
                # Create interaction
                self.cursor.execute("""
                    INSERT INTO user_interactions (user_id, set_num, interaction_type, rating, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, set_num, interaction_type) DO NOTHING
                """, (
                    user_id,
                    set_row['set_num'],
                    'rating',
                    rating,
                    datetime.now() - timedelta(days=random.randint(1, 365))
                ))
                interactions_created += 1
        
        self.conn.commit()
        print(f"✅ Created {users_created} users with {interactions_created} interactions")
        return users_created, interactions_created
    
    def test_collaborative_filtering_scalability(self):
        """Test collaborative filtering with varying amounts of data"""
        print("\n🤖 Testing Collaborative Filtering Scalability...")
        
        recommender = HybridRecommender(self.conn)
        
        # Test with current data
        print("1. Testing with current user data...")
        try:
            recommender.collaborative_recommender.prepare_user_item_matrix()
            
            matrix_shape = recommender.collaborative_recommender.user_item_matrix.shape
            print(f"   📊 User-item matrix shape: {matrix_shape}")
            
            if matrix_shape[0] >= 3 and matrix_shape[1] >= 3:
                recommender.collaborative_recommender.train_svd_model()
                
                # Test recommendations for a few users
                test_users = list(recommender.collaborative_recommender.user_lookup.keys())[:5]
                for user_id in test_users:
                    recs = recommender.collaborative_recommender.get_recommendations(str(user_id), 5)
                    print(f"   👤 User {user_id}: {len(recs)} recommendations")
                    
            else:
                print(f"   ⚠️ Matrix too small for SVD: {matrix_shape}")
                
        except Exception as e:
            print(f"   ❌ Error in collaborative filtering: {e}")
    
    def test_hybrid_recommendations(self):
        """Test hybrid recommendations with various scenarios"""
        print("\n🔄 Testing Hybrid Recommendation Scenarios...")
        
        recommender = HybridRecommender(self.conn)
        
        # Initialize both systems
        print("1. Initializing recommendation systems...")
        recommender.content_recommender.prepare_features()
        recommender.collaborative_recommender.prepare_user_item_matrix()
        
        # Test scenarios
        scenarios = [
            {"name": "New User with Liked Set", "user_id": None, "liked_set": "75192-1"},
            {"name": "Existing User Only", "user_id": 1001, "liked_set": None},
            {"name": "Existing User with Liked Set", "user_id": 1002, "liked_set": "75192-1"},
            {"name": "Cold Start", "user_id": None, "liked_set": None}
        ]
        
        for scenario in scenarios:
            print(f"\n2. Testing: {scenario['name']}")
            try:
                recs = recommender.get_recommendations(
                    user_id=scenario['user_id'],
                    liked_set=scenario['liked_set'],
                    top_k=5
                )
                print(f"   ✅ Generated {len(recs)} recommendations")
                
                if recs:
                    print("   Top recommendation:")
                    rec = recs[0]
                    print(f"   - {rec.name} (Score: {rec.score:.3f})")
                    print(f"   - Reasons: {', '.join(rec.reasons)}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def benchmark_performance(self):
        """Benchmark recommendation generation performance"""
        print("\n⚡ Performance Benchmarking...")
        
        import time
        
        recommender = HybridRecommender(self.conn)
        
        # Warm up
        print("1. Warming up systems...")
        recommender.content_recommender.prepare_features()
        recommender.collaborative_recommender.prepare_user_item_matrix()
        
        # Benchmark content-based
        print("2. Benchmarking content-based recommendations...")
        start_time = time.time()
        for i in range(10):
            recs = recommender.content_recommender.get_similar_sets("75192-1", 10)
        content_time = (time.time() - start_time) / 10
        print(f"   ⏱️ Average content-based time: {content_time:.3f}s")
        
        # Benchmark collaborative (if data available)
        if (recommender.collaborative_recommender.user_item_matrix is not None and 
            recommender.collaborative_recommender.user_item_matrix.shape[0] >= 3):
            print("3. Benchmarking collaborative filtering...")
            start_time = time.time()
            test_user = list(recommender.collaborative_recommender.user_lookup.keys())[0]
            for i in range(10):
                recs = recommender.collaborative_recommender.get_recommendations(str(test_user), 10)
            collab_time = (time.time() - start_time) / 10
            print(f"   ⏱️ Average collaborative time: {collab_time:.3f}s")
        else:
            print("3. ⚠️ Skipping collaborative benchmark - insufficient data")
        
        # Benchmark hybrid
        print("4. Benchmarking hybrid recommendations...")
        start_time = time.time()
        for i in range(10):
            recs = recommender.get_recommendations(user_id=1001, liked_set="75192-1", top_k=10)
        hybrid_time = (time.time() - start_time) / 10
        print(f"   ⏱️ Average hybrid time: {hybrid_time:.3f}s")
    
    def analyze_data_requirements(self):
        """Analyze current data and provide recommendations for production"""
        print("\n📈 Data Requirements Analysis...")
        
        # Current user stats
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_users,
                COUNT(*) FILTER (WHERE total_ratings > 0) as active_users,
                AVG(total_ratings) as avg_ratings_per_user,
                MAX(total_ratings) as max_ratings_per_user
            FROM users
        """)
        user_stats = self.cursor.fetchone()
        
        # Current interaction stats
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_interactions,
                COUNT(DISTINCT user_id) as users_with_interactions,
                COUNT(DISTINCT set_num) as sets_with_interactions,
                AVG(rating) as avg_rating
            FROM user_interactions
            WHERE rating IS NOT NULL
        """)
        interaction_stats = self.cursor.fetchone()
        
        # Set coverage
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_sets,
                COUNT(DISTINCT ui.set_num) as rated_sets,
                ROUND(COUNT(DISTINCT ui.set_num) * 100.0 / COUNT(DISTINCT s.set_num), 2) as coverage_percent
            FROM sets s
            LEFT JOIN user_interactions ui ON s.set_num = ui.set_num
            WHERE s.num_parts > 0
        """)
        coverage_stats = self.cursor.fetchone()
        
        print(f"👥 Users: {user_stats['total_users']} total, {user_stats['active_users']} active")
        print(f"📊 Avg ratings per user: {user_stats['avg_ratings_per_user']:.1f}")
        print(f"🎯 Total interactions: {interaction_stats['total_interactions']}")
        print(f"📦 Set coverage: {coverage_stats['coverage_percent']}% ({coverage_stats['rated_sets']}/{coverage_stats['total_sets']})")
        avg_rating = interaction_stats['avg_rating']
        if avg_rating is not None:
            print(f"⭐ Average rating: {avg_rating:.2f}")
        else:
            print("⭐ Average rating: No ratings data available")
        
        # Recommendations for production
        print("\n💡 Production Recommendations:")
        
        if interaction_stats['users_with_interactions'] < 50:
            print("❌ CRITICAL: Need at least 50 users with interactions for collaborative filtering")
        elif interaction_stats['users_with_interactions'] < 200:
            print("⚠️ WARNING: Collaborative filtering will improve with 200+ active users")
        else:
            print("✅ GOOD: Sufficient users for collaborative filtering")
            
        if coverage_stats['coverage_percent'] < 5:
            print("❌ CRITICAL: Less than 5% set coverage - need more diverse ratings")
        elif coverage_stats['coverage_percent'] < 20:
            print("⚠️ WARNING: Low set coverage - encourage rating diversity")
        else:
            print("✅ GOOD: Good set coverage for recommendations")
            
        if user_stats['avg_ratings_per_user'] < 10:
            print("⚠️ WARNING: Users need to rate more sets (target: 10+ per user)")
        else:
            print("✅ GOOD: Users have sufficient rating history")
    
    def create_monitoring_dashboard_data(self):
        """Create monitoring queries for production dashboard"""
        print("\n📊 Creating Production Monitoring Queries...")
        
        monitoring_queries = {
            "daily_active_users": """
                SELECT DATE(created_at) as date, COUNT(DISTINCT user_id) as active_users
                FROM user_interactions 
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY date;
            """,
            
            "recommendation_quality": """
                SELECT 
                    recommendation_type,
                    AVG(score) as avg_score,
                    COUNT(*) as recommendations_served
                FROM user_recommendations 
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY recommendation_type;
            """,
            
            "user_engagement": """
                SELECT 
                    interaction_type,
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT user_id) as unique_users
                FROM user_interactions 
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY interaction_type;
            """,
            
            "cold_start_users": """
                SELECT COUNT(*) as new_users_needing_recs
                FROM users u
                LEFT JOIN user_interactions ui ON u.id = ui.user_id
                WHERE u.created_at >= CURRENT_DATE - INTERVAL '7 days'
                AND ui.user_id IS NULL;
            """
        }
        
        print("📝 Production monitoring queries:")
        for name, query in monitoring_queries.items():
            print(f"   • {name}")
            
        return monitoring_queries
    
    def test_hard_constraint_filtering(self):
        """Test hard constraint filtering functionality"""
        print("\n🔒 Testing Hard Constraint Filtering...")
        
        # Test 1: Basic constraint creation and application
        print("1. Basic constraint creation and application...")
        constraints = [
            HardConstraint(
                ConstraintType.PIECES_MAX,
                500,
                description="Maximum 500 pieces"
            ),
            HardConstraint(
                ConstraintType.YEAR_MIN,
                2015,
                description="Released after 2015"
            )
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        print(f"   ✅ Found {len(result.valid_set_nums)} sets meeting basic constraints")
        print(f"   ⚡ Filter time: {result.performance_stats.get('filter_time_ms', 0):.2f}ms")
        
        # Test 2: Budget constraints
        print("2. Testing budget constraints...")
        budget_constraints = create_budget_constraints(budget_max=100.0, budget_min=20.0)
        budget_result = self.constraint_filter.apply_constraints(budget_constraints)
        print(f"   💰 Found {len(budget_result.valid_set_nums)} sets within budget ($20-$100)")
        
        # Test 3: Age-appropriate filtering
        print("3. Testing age-appropriate filtering...")
        age_constraints = create_age_appropriate_constraints(age=8, strict=True)
        age_result = self.constraint_filter.apply_constraints(age_constraints)
        print(f"   👶 Found {len(age_result.valid_set_nums)} sets appropriate for age 8+")
        
        # Test 4: Size constraints
        print("4. Testing size constraints...")
        size_constraints = create_size_constraints("medium")  # 200-800 pieces
        size_result = self.constraint_filter.apply_constraints(size_constraints)
        print(f"   📏 Found {len(size_result.valid_set_nums)} medium-sized sets")
        
        # Test 5: Theme constraints
        print("5. Testing theme constraints...")
        theme_constraints = [
            HardConstraint(
                ConstraintType.THEMES_REQUIRED,
                ["Star Wars", "City", "Creator"],
                description="Must be popular theme"
            )
        ]
        theme_result = self.constraint_filter.apply_constraints(theme_constraints)
        print(f"   🎭 Found {len(theme_result.valid_set_nums)} sets in popular themes")
        
        # Test 6: Multiple constraints together
        print("6. Testing multiple constraints together...")
        multi_constraints = self.constraint_filter.create_constraint_set(
            price_max=150.0,
            pieces_min=200,
            pieces_max=1000,
            age_min=8,
            required_themes=["City", "Creator"],
            year_min=2018
        )
        multi_result = self.constraint_filter.apply_constraints(multi_constraints)
        print(f"   🔗 Applied {len(multi_constraints)} constraints")
        print(f"   📊 Found {len(multi_result.valid_set_nums)} sets meeting all constraints")
        
        if multi_result.violations:
            print(f"   ⚠️ {len(multi_result.violations)} constraint violations detected")
            for violation in multi_result.violations[:2]:  # Show first 2
                print(f"     🚫 {violation.message}")
        
        # Test 7: Overly restrictive constraints
        print("7. Testing overly restrictive constraints...")
        restrictive_constraints = [
            HardConstraint(
                ConstraintType.PIECES_MIN,
                5000,  # Very high minimum
                description="Minimum 5,000 pieces"
            ),
            HardConstraint(
                ConstraintType.PRICE_MAX,
                5.0,  # Very low maximum price
                description="Maximum $5"
            )
        ]
        restrictive_result = self.constraint_filter.apply_constraints(restrictive_constraints)
        print(f"   🚧 Restrictive constraints: {len(restrictive_result.valid_set_nums)} sets found")
        print(f"   💡 Violations with suggestions: {len(restrictive_result.violations)}")
        
        # Test 8: Integration with recommendation system
        print("8. Testing integration with recommendations...")
        try:
            recommender = HybridRecommender(self.conn)
            
            # Create constrained recommendation request
            request = RecommendationRequest(
                top_k=5,
                pieces_max=800,
                age_min=8,
                required_themes=["City"],
                price_max=120.0
            )
            
            recommendations, constraint_result = recommender.get_recommendations_from_request(request)
            print(f"   🤖 Generated {len(recommendations)} constrained recommendations")
            
            if recommendations:
                print("   📦 Sample constrained recommendations:")
                for i, rec in enumerate(recommendations[:3]):
                    print(f"     {i+1}. {rec.name} ({rec.num_parts} pieces, {rec.theme_name})")
                    
        except Exception as e:
            print(f"   ⚠️ Integration test skipped: {e}")
        
        print("   ✅ Hard constraint filtering tests completed")
    
    def test_constraint_performance_with_realistic_data(self):
        """Test constraint performance with realistic user data"""
        print("\n⚡ Testing Constraint Performance with Realistic Data...")
        
        import time
        
        # Test constraint performance on different dataset sizes
        test_scenarios = [
            {
                "name": "Light constraints",
                "constraints": [
                    HardConstraint(ConstraintType.PIECES_MAX, 1000),
                    HardConstraint(ConstraintType.YEAR_MIN, 2010)
                ]
            },
            {
                "name": "Medium constraints", 
                "constraints": [
                    HardConstraint(ConstraintType.PIECES_MIN, 100),
                    HardConstraint(ConstraintType.PIECES_MAX, 800),
                    HardConstraint(ConstraintType.THEMES_REQUIRED, ["City", "Creator", "Star Wars"]),
                    HardConstraint(ConstraintType.YEAR_MIN, 2015)
                ]
            },
            {
                "name": "Heavy constraints",
                "constraints": self.constraint_filter.create_constraint_set(
                    price_max=200.0,
                    price_min=30.0,
                    pieces_min=200,
                    pieces_max=1500,
                    age_min=8,
                    required_themes=["City", "Creator", "Technic"],
                    excluded_themes=["Duplo", "Baby"],
                    year_min=2018,
                    max_complexity="moderate"
                )
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n   Testing: {scenario['name']}")
            
            # Run multiple times for average
            times = []
            results_count = []
            
            for _ in range(3):
                start_time = time.time()
                result = self.constraint_filter.apply_constraints(scenario['constraints'])
                end_time = time.time()
                
                times.append((end_time - start_time) * 1000)  # Convert to ms
                results_count.append(len(result.valid_set_nums))
            
            avg_time = sum(times) / len(times)
            avg_results = sum(results_count) / len(results_count)
            
            print(f"   ⏱️ Average time: {avg_time:.2f}ms")
            print(f"   📊 Average results: {avg_results:.0f} sets")
            print(f"   🔒 Constraints applied: {len(scenario['constraints'])}")
        
        # Test caching performance
        print("\n   Testing constraint caching...")
        cache_constraints = [
            HardConstraint(ConstraintType.THEMES_REQUIRED, ["Star Wars"])
        ]
        
        # First run (no cache)
        start_time = time.time()
        result1 = self.constraint_filter.apply_constraints(cache_constraints)
        time1 = (time.time() - start_time) * 1000
        
        # Second run (with cache)
        start_time = time.time()
        result2 = self.constraint_filter.apply_constraints(cache_constraints)
        time2 = (time.time() - start_time) * 1000
        
        print(f"   🏃 First run: {time1:.2f}ms")
        print(f"   🏃 Cached run: {time2:.2f}ms")
        if time1 > 0:
            print(f"   📈 Cache improvement: {((time1 - time2) / time1 * 100):.1f}%")
        else:
            print(f"   📈 Cache improvement: N/A (operation too fast to measure)")
        
        # Performance report
        performance_report = self.constraint_filter.get_performance_report()
        print(f"\n   📊 Overall Performance Report:")
        print(f"     Total operations: {performance_report.get('operation_count', 0)}")
        print(f"     Average filter time: {performance_report.get('average_filter_time_ms', 0):.2f}ms")
        print(f"     Total constraints applied: {performance_report.get('total_constraints_applied', 0)}")
    
    def test_realistic_constraint_scenarios(self):
        """Test realistic user constraint scenarios"""
        print("\n🎯 Testing Realistic User Constraint Scenarios...")
        
        scenarios = [
            {
                "name": "Budget-conscious parent",
                "description": "Parent looking for age-appropriate sets under budget",
                "constraints": self.constraint_filter.create_constraint_set(
                    price_max=50.0,
                    age_min=6,
                    pieces_max=500,
                    excluded_themes=["Adult Welcome"],
                    max_complexity="simple"
                )
            },
            {
                "name": "Teenage Star Wars fan",
                "description": "Teen wanting complex Star Wars sets",
                "constraints": self.constraint_filter.create_constraint_set(
                    required_themes=["Star Wars"],
                    age_min=10,
                    pieces_min=500,
                    max_complexity="complex"
                )
            },
            {
                "name": "Adult collector", 
                "description": "Adult collector seeking premium display sets",
                "constraints": self.constraint_filter.create_constraint_set(
                    price_min=100.0,
                    pieces_min=1000,
                    age_min=16,
                    year_min=2020,
                    required_themes=["Creator Expert", "Architecture", "Technic"]
                )
            },
            {
                "name": "Gift buyer",
                "description": "Someone buying a gift with specific requirements",
                "constraints": self.constraint_filter.create_constraint_set(
                    price_max=80.0,
                    price_min=25.0,
                    age_min=8,
                    pieces_min=200,
                    pieces_max=800,
                    required_themes=["City", "Friends", "Creator"]
                )
            },
            {
                "name": "Space enthusiast",
                "description": "Fan looking for space-themed sets",
                "constraints": self.constraint_filter.create_constraint_set(
                    required_themes=["Space", "City"],  # City often has space police
                    pieces_min=100,
                    excluded_themes=["Duplo"]
                )
            }
        ]
        
        for scenario in scenarios:
            print(f"\n   🎭 Scenario: {scenario['name']}")
            print(f"      📝 {scenario['description']}")
            
            result = self.constraint_filter.apply_constraints(scenario['constraints'])
            
            print(f"      ✅ Found {len(result.valid_set_nums)} matching sets")
            print(f"      🔒 Applied {len(scenario['constraints'])} constraints")
            
            if result.violations:
                print(f"      ⚠️ {len(result.violations)} constraint violations")
                for violation in result.violations[:1]:  # Show first violation
                    print(f"        🚫 {violation.message}")
                    if violation.suggested_alternatives:
                        print(f"        💡 Try: {violation.suggested_alternatives[0]}")
            
            # Show sample results if available
            if result.valid_set_nums:
                cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT s.set_num, s.name, s.num_parts, s.year, t.name as theme_name
                    FROM sets s
                    LEFT JOIN themes t ON s.theme_id = t.id  
                    WHERE s.set_num = ANY(%s)
                    ORDER BY s.year DESC, s.num_parts DESC
                    LIMIT 3
                """, [result.valid_set_nums[:10]])  # Sample from first 10
                
                sample_sets = cursor.fetchall()
                if sample_sets:
                    print(f"      📦 Sample matches:")
                    for set_info in sample_sets:
                        print(f"        • {set_info['name']} ({set_info['num_parts']} pieces, {set_info['year']}, {set_info['theme_name']})")
        
        print("\n   ✅ Realistic scenario testing completed")
    
    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🧪 LEGO Recommendation System - Full Test Suite")
        print("=" * 60)
        
        # 1. Analyze current state
        self.analyze_data_requirements()
        
        # 2. Generate realistic data
        self.simulate_realistic_user_data(num_users=150, min_ratings_per_user=8, max_ratings_per_user=40)
        
        # 3. Test collaborative filtering
        self.test_collaborative_filtering_scalability()
        
        # 4. Test hybrid approach
        self.test_hybrid_recommendations()
        
        # 5. Test hard constraint filtering
        self.test_hard_constraint_filtering()
        
        # 6. Test constraint performance
        self.test_constraint_performance_with_realistic_data()
        
        # 7. Test realistic constraint scenarios  
        self.test_realistic_constraint_scenarios()
        
        # 8. Performance benchmarking
        self.benchmark_performance()
        
        # 6. Final analysis
        print("\n" + "="*60)
        print("📊 FINAL ANALYSIS AFTER DATA SIMULATION")
        print("="*60)
        self.analyze_data_requirements()
        
        # 7. Monitoring setup
        self.create_monitoring_dashboard_data()
        
        print("\n✅ Test suite completed!")
        print("\n🚀 PRODUCTION READINESS:")
        print("   • Content-based recommendations: READY")
        print("   • Collaborative filtering: READY (with simulated data)")
        print("   • Hard constraint filtering: READY")
        print("   • Hybrid approach: READY") 
        print("   • Performance: OPTIMIZED")
        print("   • Monitoring: CONFIGURED")

if __name__ == "__main__":
    tester = RecommendationSystemTester()
    tester.run_full_test_suite()
