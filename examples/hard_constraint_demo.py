#!/usr/bin/env python3
"""
Hard Constraint Filtering Demo

This demo showcases the new hard constraint filtering system for the LEGO recommendation engine.
It demonstrates various constraint scenarios and shows how the system enforces absolute requirements.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import List

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from hard_constraint_filter import (
    HardConstraintFilter, HardConstraint, ConstraintResult,
    ConstraintType, create_budget_constraints, create_age_appropriate_constraints
)
from recommendation_system import HybridRecommender, RecommendationRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_database_connection():
    """Set up database connection"""
    # Validate required environment variables
    required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}, using defaults")
    
    db_params = {
        "host": os.getenv("DB_HOST", "localhost"),
        "database": os.getenv("DB_NAME", "brickbrain"),
        "user": os.getenv("DB_USER", "brickbrain"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", 5432))
    }
    
    if not db_params["password"]:
        raise ValueError("DB_PASSWORD environment variable is required")
    
    try:
        conn = psycopg2.connect(**db_params)
        # Test connection
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        logger.info("✅ Connected to database")
        return conn
    except psycopg2.Error as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        raise

def demo_scenario_1_budget_constraint(constraint_filter: HardConstraintFilter):
    """Demo: Budget constraint - no sets over $100"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 1: Budget Constraint - No sets over $100")
    print("="*60)
    
    constraints = create_budget_constraints(budget_max=100.0)
    result = constraint_filter.apply_constraints(constraints)
    
    print(f"💰 Applied budget constraint: Maximum $100")
    print(f"📊 Sets meeting constraint: {len(result.valid_set_nums)}")
    print(f"⏱️ Filter time: {result.performance_stats.get('filter_time_ms', 0):.2f}ms")
    
    if result.violations:
        print(f"⚠️ Constraint violations detected:")
        for violation in result.violations:
            print(f"   🚫 {violation.message}")
            if violation.suggested_alternatives:
                for alt in violation.suggested_alternatives[:2]:
                    print(f"   💡 {alt}")
    
    # Show sample results
    if result.valid_set_nums:
     # Show sample results
     if result.valid_set_nums:
         conn = constraint_filter.dbcon
         with conn.cursor(cursor_factory=RealDictCursor) as cursor:
             cursor.execute("""
                 SELECT s.set_num, s.name, s.num_parts, t.name as theme_name, s.year
                 FROM sets s
                 LEFT JOIN themes t ON s.theme_id = t.id
                 WHERE s.set_num = ANY(%s)
                 ORDER BY s.num_parts DESC
                 LIMIT 5
             """, [result.valid_set_nums])
 
             samples = cursor.fetchall()
        
        samples = cursor.fetchall()
        print("\n📦 Sample budget-friendly sets:")
        for i, row in enumerate(samples, 1):
            estimated_price = row['num_parts'] * 0.12  # Rough estimate
            print(f"   {i}. {row['name']} ({row['num_parts']} pieces, ~${estimated_price:.2f})")
            print(f"      Theme: {row['theme_name']}, Year: {row['year']}")

def demo_scenario_2_age_and_complexity(constraint_filter: HardConstraintFilter):
    """Demo: Age and complexity constraints for 8-year-old"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 2: Age-Appropriate Sets for 8-Year-Old")
    print("="*60)
    
    constraints = constraint_filter.create_constraint_set(
        age_min=8,
        age_max=12,
        max_complexity="moderate",
        pieces_max=800,  # Age-appropriate piece count
        excluded_themes=["Technic", "Architecture"]  # Too advanced
    )
    
    result = constraint_filter.apply_constraints(constraints)
    
    print(f"👶 Applied age constraints: 8-12 years old")
    print(f"🧩 Complexity: Moderate or simpler")
    print(f"🔢 Pieces: Maximum 800")
    print(f"🚫 Excluded: Technic, Architecture (too advanced)")
    print(f"📊 Sets meeting all constraints: {len(result.valid_set_nums)}")
    
    # Show themed breakdown
    if result.valid_set_nums:
        conn = constraint_filter.dbcon
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT t.name as theme_name, COUNT(*) as count
            FROM sets s
            LEFT JOIN themes t ON s.theme_id = t.id
            WHERE s.set_num = ANY(%s)
            GROUP BY t.name
            ORDER BY count DESC
            LIMIT 5
        """, [result.valid_set_nums])
        
        themes = cursor.fetchall()
        print("\n🎨 Top themes for 8-year-olds:")
        for theme in themes:
            print(f"   • {theme['theme_name']}: {theme['count']} sets")

def demo_scenario_3_star_wars_collection(constraint_filter: HardConstraintFilter):
    """Demo: Star Wars collection with specific requirements"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 3: Star Wars Collection - Specific Requirements")
    print("="*60)
    
    constraints = constraint_filter.create_constraint_set(
        required_themes=["Star Wars"],
        pieces_min=300,  # Substantial builds
        pieces_max=2000,  # Not too overwhelming
        year_min=2015,  # Modern sets only
        price_max=200.0  # Budget limit
    )
    
    result = constraint_filter.apply_constraints(constraints)
    
    print(f"⭐ Required theme: Star Wars only")
    print(f"🔢 Pieces: 300-2000 (substantial but manageable)")
    print(f"📅 Year: 2015 or newer (modern sets)")
    print(f"💰 Budget: Under $200")
    print(f"📊 Star Wars sets meeting criteria: {len(result.valid_set_nums)}")
    
    if result.valid_set_nums:
        conn = constraint_filter.dbcon
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT s.set_num, s.name, s.num_parts, s.year
            FROM sets s
            LEFT JOIN themes t ON s.theme_id = t.id
            WHERE s.set_num = ANY(%s)
            ORDER BY s.num_parts DESC
            LIMIT 7
        """, [result.valid_set_nums])
        
        sets = cursor.fetchall()
        print("\n🚀 Star Wars sets meeting all constraints:")
        for i, row in enumerate(sets, 1):
            estimated_price = row['num_parts'] * 0.12
            print(f"   {i}. {row['name']}")
            print(f"      {row['num_parts']} pieces, {row['year']}, ~${estimated_price:.2f}")

def demo_scenario_4_overly_restrictive(constraint_filter: HardConstraintFilter):
    """Demo: Overly restrictive constraints with violation feedback"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 4: Overly Restrictive Constraints (Violation Demo)")
    print("="*60)
    
    # Create impossible constraints
    constraints = constraint_filter.create_constraint_set(
        price_max=5.0,  # Impossibly low
        pieces_min=5000,  # Impossibly high
        required_themes=["Duplo"],  # Conflicting with piece count
        year_min=2025  # Future sets
    )
    
    result = constraint_filter.apply_constraints(constraints)
    
    print(f"💸 Maximum price: $5 (impossibly low)")
    print(f"🔢 Minimum pieces: 5000 (impossibly high)")
    print(f"🧸 Required theme: Duplo (conflicts with piece count)")
    print(f"📅 Year: 2025+ (future sets)")
    print(f"📊 Sets meeting ALL constraints: {len(result.valid_set_nums)}")
    
    print(f"\n⚠️ Constraint violations detected: {len(result.violations)}")
    for i, violation in enumerate(result.violations, 1):
        elimination_rate = violation.violating_count / max(violation.total_count, 1) * 100
        print(f"\n   {i}. {violation.constraint.description}")
        print(f"      📉 Eliminated {violation.violating_count}/{violation.total_count} sets ({elimination_rate:.1f}%)")
        print(f"      💬 {violation.message}")
        
        if violation.suggested_alternatives:
            print(f"      💡 Suggestions:")
            for alt in violation.suggested_alternatives:
                print(f"         • {alt}")

def demo_scenario_5_hybrid_recommendations(hybrid_recommender: HybridRecommender):
    """Demo: Integration with hybrid recommendation system"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 5: Hybrid Recommendations with Hard Constraints")
    print("="*60)
    
    # Create a recommendation request with constraints
    request = RecommendationRequest(
        top_k=5,
        pieces_min=200,
        pieces_max=1500,
        age_min=10,
        required_themes=["City", "Creator", "Friends"],
        max_complexity="moderate",
        exclude_owned=False
    )
    
    try:
        try:
            recommendations, constraint_result = hybrid_recommender.get_recommendations_from_request(request)
            
            print(f"🎯 Recommendation request with constraints:")
            print(f"   🔢 Pieces: 200-1500")
            print(f"   👶 Age: 10+")
            print(f"   🎨 Themes: City, Creator, Friends")
            print(f"   🧩 Complexity: Moderate or simpler")
            
            print(f"\n📊 Constraint application results:")
            if constraint_result:
                print(f"   🔒 Constraints applied: {len(constraint_result.applied_constraints)}")
                print(f"   ✅ Valid sets found: {len(constraint_result.valid_set_nums)}")
                print(f"   ⏱️ Filter time: {constraint_result.performance_stats.get('filter_time_ms', 0):.2f}ms")
            
            print(f"\n🎉 Final recommendations: {len(recommendations)}")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec.name}")
                print(f"      Theme: {rec.theme_name}, {rec.num_parts} pieces, {rec.year}")
                print(f"      Score: {rec.score:.3f}")
                print(f"      Reasons: {', '.join(rec.reasons[:2])}")
                
        except (AttributeError, ValueError) as e:
            print(f"❌ Hybrid recommendation demo failed: {e}")
            logger.error(f"Hybrid recommendation error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in hybrid recommendation demo: {e}")
            logger.error(f"Unexpected hybrid recommendation error: {e}")
            raise  # Re-raise unexpected errors for debugging

def demo_performance_comparison(constraint_filter: HardConstraintFilter):
    """Demo: Performance comparison of different constraint types"""
    print("\n" + "="*60)
    print("🎯 SCENARIO 6: Performance Comparison")
    print("="*60)
    
    scenarios = [
        ("Simple piece count", [HardConstraint(ConstraintType.PIECES_MAX, 500)]),
        ("Theme constraint", [HardConstraint(ConstraintType.THEMES_REQUIRED, ["City"])]),
        ("Multiple constraints", [
            HardConstraint(ConstraintType.PIECES_MIN, 100),
            HardConstraint(ConstraintType.PIECES_MAX, 800),
            HardConstraint(ConstraintType.YEAR_MIN, 2018)
        ]),
        ("Complex constraints", [
            HardConstraint(ConstraintType.PIECES_MIN, 200),
            HardConstraint(ConstraintType.PIECES_MAX, 1000),
            HardConstraint(ConstraintType.THEMES_REQUIRED, ["Star Wars", "City"]),
            HardConstraint(ConstraintType.YEAR_MIN, 2015),
            HardConstraint(ConstraintType.PRICE_MAX, 150.0)
        ])
    ]
    
    print("⚡ Performance comparison:")
    for name, constraints in scenarios:
        result = constraint_filter.apply_constraints(constraints)
        filter_time = result.performance_stats.get('filter_time_ms', 0)
        print(f"   {name:20} | {len(constraints):2} constraints | {len(result.valid_set_nums):4} results | {filter_time:6.2f}ms")
    
    # Overall performance stats
    performance_report = constraint_filter.get_performance_report()
    print(f"\n📊 Overall performance statistics:")
    print(f"   Total constraints applied: {performance_report['total_constraints_applied']}")
    print(f"   Total sets filtered: {performance_report['total_sets_filtered']}")
    print(f"   Average filter time: {performance_report['average_filter_time_ms']:.2f}ms")
    print(f"   Total operations: {performance_report.get('operation_count', 0)}")

def main():
    """Run the hard constraint filtering demo"""
    print("🧱 LEGO Hard Constraint Filtering System Demo")
    print("=" * 60)
    print("This demo showcases the new hard constraint filtering system")
    print("that ensures ALL recommendations meet absolute requirements.")
    print()
    
    try:
        # Setup
        conn = setup_database_connection()
        constraint_filter = HardConstraintFilter(conn)
        hybrid_recommender = HybridRecommender(conn)
        
        # Check database has data
        # Check database has data
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sets WHERE num_parts > 0")
            set_count = cursor.fetchone()[0]
        
        if set_count == 0:
            print("❌ No LEGO sets found in database. Please load data first.")
            return
        
        print(f"✅ Found {set_count} LEGO sets in database")
        
        # Run demo scenarios
        demo_scenario_1_budget_constraint(constraint_filter)
        demo_scenario_2_age_and_complexity(constraint_filter)  
        demo_scenario_3_star_wars_collection(constraint_filter)
        demo_scenario_4_overly_restrictive(constraint_filter)
        demo_scenario_5_hybrid_recommendations(hybrid_recommender)
        demo_performance_comparison(constraint_filter)
        
        print("\n" + "="*60)
        print("🎉 Hard Constraint Filtering Demo Complete!")
        print("="*60)
        print()
        print("Key Benefits Demonstrated:")
        print("✅ Absolute constraint enforcement - no fuzzy matching")
        print("✅ Multiple constraint types supported")
        print("✅ Intelligent violation detection and suggestions")
        print("✅ Integration with existing recommendation system")
        print("✅ Performance optimized with caching")
        print("✅ Clear feedback when constraints can't be met")
        print()
        print("🚀 The system is ready for production use!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Please check database connection and data availability.")
    
    finally:
        if 'conn' in locals():
            try:
                conn.close()
                logger.info("✅ Database connection closed successfully")
            except Exception as close_error:
                logger.error(f"⚠️ Error closing database connection: {close_error}")
                # Don't re-raise the exception to avoid masking the original error

if __name__ == "__main__":
    main()
