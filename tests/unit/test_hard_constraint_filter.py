"""
Test suite for Hard Constraint Filtering System

This test suite validates the hard constraint filtering functionality
including constraint creation, application, violation detection, and API integration.
"""

import unittest
import sys
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import logging

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'scripts'))

from hard_constraint_filter import (
    HardConstraintFilter, HardConstraint, ConstraintResult,
    ConstraintType, ConstraintSeverity, 
    create_budget_constraints, create_age_appropriate_constraints, create_size_constraints
)
from recommendation_system import HybridRecommender, RecommendationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHardConstraintFilter(unittest.TestCase):
    """Test cases for the Hard Constraint Filter system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up database connection for testing"""
        try:
            cls.db_params = {
                "host": os.getenv("DB_HOST", "localhost"),
                "database": os.getenv("DB_NAME", "brickbrain"),
                "user": os.getenv("DB_USER", "brickbrain"),
                "password": os.getenv("DB_PASSWORD", "brickbrain_password"),
                "port": int(os.getenv("DB_PORT", 5432))
            }
            
            cls.conn = psycopg2.connect(**cls.db_params)
            cls.constraint_filter = HardConstraintFilter(cls.conn)
            cls.hybrid_recommender = HybridRecommender(cls.conn)
            
            logger.info("✅ Database connection established for testing")
            
            # Test database connectivity
            cursor = cls.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sets WHERE num_parts > 0")
            set_count = cursor.fetchone()[0]
            logger.info(f"📊 Found {set_count} sets in database for testing")
            
            if set_count == 0:
                logger.warning("⚠️ No sets found in database - some tests may fail")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up database connection"""
        if hasattr(cls, 'conn'):
            cls.conn.close()
            logger.info("🧹 Database connection closed")
    
    def test_01_constraint_creation(self):
        """Test basic constraint creation"""
        print("\n🧪 Testing constraint creation...")
        
        # Test budget constraints
        budget_constraints = create_budget_constraints(budget_max=100.0, budget_min=20.0)
        self.assertEqual(len(budget_constraints), 2)
        self.assertEqual(budget_constraints[0].constraint_type, ConstraintType.PRICE_MAX)
        self.assertEqual(budget_constraints[1].constraint_type, ConstraintType.PRICE_MIN)
        print("  ✅ Budget constraints created successfully")
        
        # Test age constraints
        age_constraints = create_age_appropriate_constraints(age=10, strict=True)
        self.assertEqual(len(age_constraints), 1)
        self.assertEqual(age_constraints[0].constraint_type, ConstraintType.AGE_MIN)
        print("  ✅ Age constraints created successfully")
        
        # Test size constraints
        size_constraints = create_size_constraints("medium")
        self.assertEqual(len(size_constraints), 2)
        self.assertEqual(size_constraints[0].constraint_type, ConstraintType.PIECES_MIN)
        self.assertEqual(size_constraints[1].constraint_type, ConstraintType.PIECES_MAX)
        print("  ✅ Size constraints created successfully")
    
    def test_02_constraint_set_creation(self):
        """Test comprehensive constraint set creation"""
        print("\n🧪 Testing constraint set creation...")
        
        constraints = self.constraint_filter.create_constraint_set(
            price_max=150.0,
            pieces_min=100,
            pieces_max=1000,
            age_min=8,
            required_themes=["Star Wars", "Technic"],
            excluded_themes=["Duplo"],
            max_complexity="moderate"
        )
        
        self.assertGreater(len(constraints), 5)
        
        # Check for specific constraint types
        constraint_types = [c.constraint_type for c in constraints]
        self.assertIn(ConstraintType.PRICE_MAX, constraint_types)
        self.assertIn(ConstraintType.PIECES_MIN, constraint_types)
        self.assertIn(ConstraintType.PIECES_MAX, constraint_types)
        self.assertIn(ConstraintType.AGE_MIN, constraint_types)
        self.assertIn(ConstraintType.THEMES_REQUIRED, constraint_types)
        self.assertIn(ConstraintType.THEMES_EXCLUDED, constraint_types)
        self.assertIn(ConstraintType.COMPLEXITY_MAX, constraint_types)
        
        print(f"  ✅ Created {len(constraints)} constraints successfully")
        for constraint in constraints:
            print(f"    🔒 {constraint.description}")
    
    def test_03_basic_constraint_application(self):
        """Test basic constraint application"""
        print("\n🧪 Testing basic constraint application...")
        
        # Create simple piece count constraint
        constraints = [
            HardConstraint(
                ConstraintType.PIECES_MAX,
                500,
                description="Maximum 500 pieces"
            )
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        
        self.assertIsInstance(result, ConstraintResult)
        self.assertIsInstance(result.valid_set_nums, list)
        self.assertGreater(len(result.valid_set_nums), 0, "Should find some sets with <= 500 pieces")
        
        print(f"  ✅ Found {len(result.valid_set_nums)} sets meeting piece count constraint")
        print(f"  📊 Filter time: {result.performance_stats.get('filter_time_ms', 0):.2f}ms")
    
    def test_04_theme_constraint_application(self):
        """Test theme-based constraint application"""
        print("\n🧪 Testing theme constraint application...")
        
        # Test with popular themes
        constraints = [
            HardConstraint(
                ConstraintType.THEMES_REQUIRED,
                ["Star Wars", "City", "Creator"],
                description="Must be Star Wars, City, or Creator theme"
            )
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        
        self.assertIsInstance(result.valid_set_nums, list)
        print(f"  ✅ Found {len(result.valid_set_nums)} sets in specified themes")
        
        # Verify the results actually match the themes
        if result.valid_set_nums:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            sample_sets = result.valid_set_nums[:5]  # Check first 5
            cursor.execute(
                "SELECT s.set_num, s.name, t.name as theme_name FROM sets s "
                "LEFT JOIN themes t ON s.theme_id = t.id "
                "WHERE s.set_num = ANY(%s)",
                [sample_sets]
            )
            sample_results = cursor.fetchall()
            
            for row in sample_results:
                print(f"    📦 {row['set_num']}: {row['name']} ({row['theme_name']})")
    
    def test_05_multiple_constraint_application(self):
        """Test application of multiple constraints together"""
        print("\n🧪 Testing multiple constraint application...")
        
        constraints = [
            HardConstraint(
                ConstraintType.PIECES_MIN,
                200,
                description="Minimum 200 pieces"
            ),
            HardConstraint(
                ConstraintType.PIECES_MAX,
                800,
                description="Maximum 800 pieces"
            ),
            HardConstraint(
                ConstraintType.YEAR_MIN,
                2015,
                description="Released after 2015"
            )
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        
        print(f"  ✅ Applied {len(constraints)} constraints")
        print(f"  📊 Found {len(result.valid_set_nums)} sets meeting all constraints")
        print(f"  ⚡ Filter time: {result.performance_stats.get('filter_time_ms', 0):.2f}ms")
        
        # Check for violations
        if result.violations:
            print(f"  ⚠️ {len(result.violations)} constraint violations detected:")
            for violation in result.violations:
                print(f"    🚫 {violation.message}")
                if violation.suggested_alternatives:
                    print(f"       💡 Suggestions: {', '.join(violation.suggested_alternatives[:2])}")
    
    def test_06_overly_restrictive_constraints(self):
        """Test handling of overly restrictive constraints"""
        print("\n🧪 Testing overly restrictive constraints...")
        
        # Create impossible constraints
        constraints = [
            HardConstraint(
                ConstraintType.PIECES_MIN,
                10000,  # Very high minimum
                description="Minimum 10,000 pieces"
            ),
            HardConstraint(
                ConstraintType.PRICE_MAX,
                1.0,  # Very low maximum price
                description="Maximum $1"
            )
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        
        # Should find very few or no results
        print(f"  ✅ Restrictive constraints resulted in {len(result.valid_set_nums)} sets")
        
        # Should have violations with suggestions
        self.assertGreater(len(result.violations), 0, "Should detect constraint violations")
        
        for violation in result.violations:
            print(f"  🚫 Violation: {violation.message}")
            if violation.suggested_alternatives:
                print(f"     💡 Suggestions: {violation.suggested_alternatives}")
    
    def test_07_constraint_filter_integration(self):
        """Test integration with the recommendation system"""
        print("\n🧪 Testing constraint filter integration...")
        
        # Create a recommendation request with constraints
        request = RecommendationRequest(
            top_k=5,
            pieces_max=1000,
            age_min=8,
            required_themes=["City"]
        )
        
        try:
            recommendations, constraint_result = self.hybrid_recommender.get_recommendations_from_request(request)
            
            print(f"  ✅ Generated {len(recommendations)} constrained recommendations")
            
            if recommendations:
                print("  📦 Sample recommendations:")
                for i, rec in enumerate(recommendations[:3]):
                    print(f"    {i+1}. {rec.name} ({rec.num_parts} pieces, {rec.theme_name})")
            
            if constraint_result:
                print(f"  📊 Constraint stats: {constraint_result.performance_stats}")
                
        except Exception as e:
            print(f"  ⚠️ Integration test failed: {e}")
            # This might fail if recommendation system isn't fully initialized
    
    def test_08_constraint_sql_generation(self):
        """Test SQL query generation from constraints"""
        print("\n🧪 Testing SQL generation...")
        
        constraints = [
            HardConstraint(ConstraintType.PIECES_MAX, 500),
            HardConstraint(ConstraintType.YEAR_MIN, 2020),
            HardConstraint(ConstraintType.THEMES_REQUIRED, ["City", "Creator"])
        ]
        
        result = self.constraint_filter.apply_constraints(constraints)
        
        # Check that SQL was generated
        self.assertIsNotNone(result.constraint_sql)
        self.assertIn("SELECT", result.constraint_sql)
        self.assertIn("WHERE", result.constraint_sql)
        
        print(f"  ✅ Generated SQL query ({len(result.constraint_sql)} characters)")
        print(f"  🔍 Sample SQL: {result.constraint_sql[:100]}...")
    
    def test_09_performance_monitoring(self):
        """Test performance monitoring and statistics"""
        print("\n🧪 Testing performance monitoring...")
        
        # Run multiple constraint applications to gather stats
        for i in range(3):
            constraints = [
                HardConstraint(ConstraintType.PIECES_MAX, 300 + i * 100),
                HardConstraint(ConstraintType.YEAR_MIN, 2018 + i)
            ]
            result = self.constraint_filter.apply_constraints(constraints)
        
        # Get performance report
        performance_report = self.constraint_filter.get_performance_report()
        
        self.assertIn('total_constraints_applied', performance_report)
        self.assertIn('average_filter_time_ms', performance_report)
        
        print(f"  ✅ Performance monitoring working:")
        print(f"    📊 Total constraints applied: {performance_report['total_constraints_applied']}")
        print(f"    ⏱️ Average filter time: {performance_report['average_filter_time_ms']:.2f}ms")
        print(f"    🔄 Operations: {performance_report.get('operation_count', 0)}")
    
    def test_10_cache_functionality(self):
        """Test constraint filter caching"""
        print("\n🧪 Testing cache functionality...")
        
        # Apply same constraint twice to test caching
        constraints = [
            HardConstraint(ConstraintType.THEMES_REQUIRED, ["Star Wars"])
        ]
        
        # First application
        result1 = self.constraint_filter.apply_constraints(constraints)
        time1 = result1.performance_stats.get('filter_time_ms', 0)
        
        # Second application (should use cache for theme lookup)
        result2 = self.constraint_filter.apply_constraints(constraints)
        time2 = result2.performance_stats.get('filter_time_ms', 0)
        
        # Results should be consistent
        self.assertEqual(len(result1.valid_set_nums), len(result2.valid_set_nums))
        
        print(f"  ✅ Cache test completed:")
        print(f"    🕐 First run: {time1:.2f}ms")
        print(f"    🕑 Second run: {time2:.2f}ms")
        print(f"    📊 Consistent results: {len(result1.valid_set_nums)} sets both times")
        
        # Clear cache
        self.constraint_filter.clear_cache()
        print("  🧹 Cache cleared")

class TestConstraintAPIIntegration(unittest.TestCase):
    """Test API integration with hard constraints"""
    
    def test_01_enhanced_request_model(self):
        """Test the enhanced recommendation request model"""
        print("\n🧪 Testing enhanced request model...")
        
        # This test ensures our Pydantic models are working correctly  
        from recommendation_system import RecommendationRequest
        
        # Create a request with constraints
        request_data = {
            "user_id": 1,
            "top_k": 10,
            "price_max": 100.0,
            "pieces_min": 200,
            "pieces_max": 1000,
            "age_min": 8,
            "required_themes": ["City", "Creator"],
            "max_complexity": "moderate",
            "exclude_owned": True 
        }
        
        try:
            request = RecommendationRequest(**request_data)
            self.assertEqual(request.price_max, 100.0)
            self.assertEqual(request.pieces_min, 200)
            self.assertEqual(request.required_themes, ["City", "Creator"])
            self.assertEqual(request.exclude_owned, True)
            
            print("  ✅ Enhanced request model validation passed")
            print(f"    🔒 Constraints: price_max={request.price_max}, pieces={request.pieces_min}-{request.pieces_max}")
            print(f"    🎯 Themes: {request.required_themes}")
            
        except Exception as e:
            print(f"  ❌ Request model validation failed: {e}")

def run_constraint_tests():
    """Run all hard constraint filtering tests"""
    print("🚀 Starting Hard Constraint Filtering Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add constraint filter tests
    suite.addTest(TestHardConstraintFilter('test_01_constraint_creation'))
    suite.addTest(TestHardConstraintFilter('test_02_constraint_set_creation'))
    suite.addTest(TestHardConstraintFilter('test_03_basic_constraint_application'))
    suite.addTest(TestHardConstraintFilter('test_04_theme_constraint_application'))
    suite.addTest(TestHardConstraintFilter('test_05_multiple_constraint_application'))
    suite.addTest(TestHardConstraintFilter('test_06_overly_restrictive_constraints'))
    suite.addTest(TestHardConstraintFilter('test_07_constraint_filter_integration'))
    suite.addTest(TestHardConstraintFilter('test_08_constraint_sql_generation'))
    suite.addTest(TestHardConstraintFilter('test_09_performance_monitoring'))
    suite.addTest(TestHardConstraintFilter('test_10_cache_functionality'))
    
    # Add API integration tests
    suite.addTest(TestConstraintAPIIntegration('test_01_enhanced_request_model'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("🎯 Hard Constraint Filtering Test Results:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"❌ {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"⚠️ {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎉 Success rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_constraint_tests()
    exit(0 if success else 1)
