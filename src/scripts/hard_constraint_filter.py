import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of hard constraints supported"""
    PRICE_MAX = "price_max"
    PRICE_MIN = "price_min" 
    PIECES_MAX = "pieces_max"
    PIECES_MIN = "pieces_min"
    AGE_MIN = "age_min"
    AGE_MAX = "age_max"
    YEAR_MIN = "year_min"
    YEAR_MAX = "year_max"
    THEMES_REQUIRED = "themes_required"
    THEMES_EXCLUDED = "themes_excluded"
    COMPLEXITY_MAX = "complexity_max"
    COMPLEXITY_MIN = "complexity_min"
    AVAILABILITY = "availability"
    EXCLUDE_OWNED = "exclude_owned"
    EXCLUDE_WISHLISTED = "exclude_wishlisted"

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations"""
    BLOCKING = "blocking"    # Completely prevents recommendations
    WARNING = "warning"      # Logged but doesn't block
    INFO = "info"           # Informational only

@dataclass
class HardConstraint:
    """Individual hard constraint definition"""
    constraint_type: ConstraintType
    value: Any
    severity: ConstraintSeverity = ConstraintSeverity.BLOCKING
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.constraint_type.value}: {self.value}"

@dataclass 
class ConstraintViolation:
    """Details about a constraint violation"""
    constraint: HardConstraint
    violating_count: int
    total_count: int
    message: str
    suggested_alternatives: List[str] = None

@dataclass
class ConstraintResult:
    """Result of applying hard constraints"""
    valid_set_nums: List[str]
    violations: List[ConstraintViolation] 
    applied_constraints: List[HardConstraint]
    performance_stats: Dict[str, Any]
    constraint_sql: str = ""

class HardConstraintFilter:
    """
    Main class for applying hard constraints to LEGO set recommendations
    
    This filter system ensures that returned recommendations meet ALL specified
    hard constraints. No fuzzy matching or soft preferences - constraints are absolute.
    """
    
    def __init__(self, dbcon):
        """
        Initialize the hard constraint filter
        
        Args:
            dbcon: Database connection object
        """
        self.dbcon = dbcon
        
        # Cache for constraint validation
        self._constraint_cache = {}
        self._theme_cache = {}
        self._set_metadata_cache = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_constraints_applied': 0,
            'total_sets_filtered': 0,
            'average_filter_time_ms': 0,
            'constraint_hit_rates': {}
        }
    
    def create_constraint_set(self, 
                            price_max: Optional[float] = None,
                            price_min: Optional[float] = None,
                            pieces_max: Optional[int] = None,
                            pieces_min: Optional[int] = None,
                            age_min: Optional[int] = None,
                            age_max: Optional[int] = None,
                            year_min: Optional[int] = None,
                            year_max: Optional[int] = None,
                            required_themes: Optional[List[str]] = None,
                            excluded_themes: Optional[List[str]] = None,
                            max_complexity: Optional[str] = None,
                            min_complexity: Optional[str] = None,
                            must_be_available: bool = False,
                            exclude_owned: bool = False,
                            exclude_wishlisted: bool = False,
                            user_id: Optional[int] = None) -> List[HardConstraint]:
        """
        Create a set of hard constraints from parameters
        
        Args:
            price_max: Maximum price in USD (estimated from piece count)
            price_min: Minimum price in USD (estimated from piece count)  
            pieces_max: Maximum number of pieces
            pieces_min: Minimum number of pieces
            age_min: Minimum recommended age
            age_max: Maximum recommended age
            year_min: Earliest release year
            year_max: Latest release year
            required_themes: List of theme names that must be included
            excluded_themes: List of theme names that must be excluded
            max_complexity: Maximum complexity level ('simple', 'moderate', 'complex')
            min_complexity: Minimum complexity level ('simple', 'moderate', 'complex')
            must_be_available: Whether sets must be currently available
            exclude_owned: Exclude sets owned by user (requires user_id)
            exclude_wishlisted: Exclude sets in user's wishlist (requires user_id)
            user_id: User ID for personal constraints
            
        Returns:
            List of HardConstraint objects
        """
        constraints = []
        
        # Price constraints
        if price_max is not None:
            constraints.append(HardConstraint(
                ConstraintType.PRICE_MAX, 
                price_max,
                description=f"Must cost less than ${price_max:.2f}"
            ))
            
        if price_min is not None:
            constraints.append(HardConstraint(
                ConstraintType.PRICE_MIN,
                price_min, 
                description=f"Must cost more than ${price_min:.2f}"
            ))
        
        # Piece count constraints  
        if pieces_max is not None:
            constraints.append(HardConstraint(
                ConstraintType.PIECES_MAX,
                pieces_max,
                description=f"Must have fewer than {pieces_max} pieces"
            ))
            
        if pieces_min is not None:
            constraints.append(HardConstraint(
                ConstraintType.PIECES_MIN,
                pieces_min,
                description=f"Must have more than {pieces_min} pieces" 
            ))
        
        # Age constraints
        if age_min is not None:
            constraints.append(HardConstraint(
                ConstraintType.AGE_MIN,
                age_min,
                description=f"Must be suitable for ages {age_min}+"
            ))
            
        if age_max is not None:
            constraints.append(HardConstraint(
                ConstraintType.AGE_MAX, 
                age_max,
                description=f"Must be suitable for ages up to {age_max}"
            ))
        
        # Year constraints
        if year_min is not None:
            constraints.append(HardConstraint(
                ConstraintType.YEAR_MIN,
                year_min,
                description=f"Must be released after {year_min}"
            ))
            
        if year_max is not None:
            constraints.append(HardConstraint(
                ConstraintType.YEAR_MAX,
                year_max, 
                description=f"Must be released before {year_max}"
            ))
        
        # Theme constraints
        if required_themes:
            constraints.append(HardConstraint(
                ConstraintType.THEMES_REQUIRED,
                required_themes,
                description=f"Must be from themes: {', '.join(required_themes)}"
            ))
            
        if excluded_themes:
            constraints.append(HardConstraint(
                ConstraintType.THEMES_EXCLUDED,
                excluded_themes,
                description=f"Must NOT be from themes: {', '.join(excluded_themes)}"
            ))
        
        # Complexity constraints
        if max_complexity:
            constraints.append(HardConstraint(
                ConstraintType.COMPLEXITY_MAX,
                max_complexity,
                description=f"Must be {max_complexity} complexity or simpler"
            ))
            
        if min_complexity:
            constraints.append(HardConstraint(
                ConstraintType.COMPLEXITY_MIN,
                min_complexity,
                description=f"Must be {min_complexity} complexity or more complex"
            ))
        
        # User-specific constraints
        if exclude_owned and user_id:
            constraints.append(HardConstraint(
                ConstraintType.EXCLUDE_OWNED,
                user_id,
                description="Must not be in user's collection"
            ))
            
        if exclude_wishlisted and user_id:
            constraints.append(HardConstraint(
                ConstraintType.EXCLUDE_WISHLISTED,
                user_id,
                description="Must not be in user's wishlist"
            ))
        
        # Availability constraint
        if must_be_available:
            constraints.append(HardConstraint(
                ConstraintType.AVAILABILITY,
                True,
                description="Must be currently available for purchase"
            ))
        
        return constraints
    
    def apply_constraints(self, 
                         constraints: List[HardConstraint],
                         candidate_set_nums: Optional[List[str]] = None) -> ConstraintResult:
        """
        Apply hard constraints to filter LEGO sets
        
        Args:
            constraints: List of hard constraints to apply
            candidate_set_nums: Optional list of set numbers to filter (if None, filters all sets)
            
        Returns:
            ConstraintResult with valid sets and violation details
        """
        start_time = datetime.now()
        
        logger.info(f"🔒 Applying {len(constraints)} hard constraints...")
        
        # Build SQL query with constraints
        sql_parts = self._build_constraint_sql(constraints, candidate_set_nums)
        
        try:
            cursor = self.dbcon.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql_parts['query'], sql_parts['params'])
            results = cursor.fetchall()
            
            valid_set_nums = [row['set_num'] for row in results]
            
            # Check for violations and gather stats
            violations = self._check_constraint_violations(constraints, candidate_set_nums)
            
            # Update performance stats
            filter_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(len(constraints), len(valid_set_nums), filter_time)
            
            result = ConstraintResult(
                valid_set_nums=valid_set_nums,
                violations=violations,
                applied_constraints=constraints,
                performance_stats={
                    'filter_time_ms': filter_time,
                    'input_sets': len(candidate_set_nums) if candidate_set_nums else 'all',
                    'output_sets': len(valid_set_nums),
                    'constraint_count': len(constraints)
                },
                constraint_sql=sql_parts['query']
            )
            
            logger.info(f"✅ Hard constraints applied: {len(valid_set_nums)} sets remain")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error applying hard constraints: {e}")
            raise
    
    def _build_constraint_sql(self, 
                             constraints: List[HardConstraint], 
                             candidate_set_nums: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build SQL query with constraint conditions
        
        Args:
            constraints: List of constraints to apply
            candidate_set_nums: Optional list of candidate sets
            
        Returns:
            Dict with 'query' and 'params' keys
        """
        # Base query
        base_query = """
        SELECT DISTINCT s.set_num, s.name, s.year, s.theme_id, s.num_parts, 
               t.name as theme_name, s.img_url
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        """
        
        where_conditions = []
        params = []
        
        # Always exclude sets with 0 parts (accessories/catalogs)
        where_conditions.append("s.num_parts > 0")
        
        # If candidate sets specified, limit to those
        if candidate_set_nums:
            where_conditions.append(f"s.set_num = ANY(%s)")
            params.append(candidate_set_nums)
        
        # Apply each constraint
        for constraint in constraints:
            condition, constraint_params = self._constraint_to_sql(constraint)
            if condition:
                where_conditions.append(condition)
                params.extend(constraint_params)
        
        # Combine conditions
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        full_query = f"{base_query} WHERE {where_clause} ORDER BY s.set_num"
        
        return {
            'query': full_query,
            'params': params
        }
    
    def _constraint_to_sql(self, constraint: HardConstraint) -> Tuple[str, List[Any]]:
        """
        Convert a single constraint to SQL condition
        
        Args:
            constraint: Hard constraint to convert
            
        Returns:
            Tuple of (SQL condition string, parameters list)
        """
        constraint_type = constraint.constraint_type
        value = constraint.value
        
        if constraint_type == ConstraintType.PIECES_MAX:
            return "s.num_parts <= %s", [value]
            
        elif constraint_type == ConstraintType.PIECES_MIN:
            return "s.num_parts >= %s", [value]
            
        elif constraint_type == ConstraintType.YEAR_MIN:
            return "s.year >= %s", [value]
            
        elif constraint_type == ConstraintType.YEAR_MAX:
            return "s.year <= %s", [value]
            
        elif constraint_type == ConstraintType.PRICE_MAX:
            # Estimate price from piece count (rough approximation)
            # Typical LEGO pricing: $0.10-0.15 per piece, we'll use upper bound for max constraint
            estimated_pieces = int(value / 0.10)  # Conservative estimate
            return "s.num_parts <= %s", [estimated_pieces]
            
        elif constraint_type == ConstraintType.PRICE_MIN:
            # Estimate minimum pieces needed for price
            estimated_pieces = int(value / 0.15)  # Liberal estimate for minimum
            return "s.num_parts >= %s", [estimated_pieces]
            
        elif constraint_type == ConstraintType.THEMES_REQUIRED:
            # Get theme IDs for theme names
            theme_ids = self._get_theme_ids(value)
            if theme_ids:
                return "s.theme_id = ANY(%s)", [theme_ids]
            else:
                # No matching themes found - constraint cannot be satisfied
                return "1=0", []  # Always false condition
                
        elif constraint_type == ConstraintType.THEMES_EXCLUDED:
            # Get theme IDs for excluded themes
            theme_ids = self._get_theme_ids(value)
            if theme_ids:
                return "s.theme_id NOT IN %s", [tuple(theme_ids)]
            else:
                # No matching themes to exclude - no constraint needed
                return "", []
                
        elif constraint_type == ConstraintType.AGE_MIN:
            # For age constraints, we need to estimate based on piece count and complexity
            # This is an approximation since LEGO doesn't store exact age recommendations in our DB
            if value <= 4:
                return "s.num_parts <= 50", []  # Duplo/simple sets
            elif value <= 8:
                return "s.num_parts <= 500", []  # Youth sets
            elif value <= 12:
                return "s.num_parts <= 1500", []  # Teen sets
            else:
                return "s.num_parts >= 500", []  # Adult sets
                
        elif constraint_type == ConstraintType.AGE_MAX:
            # Maximum age constraint (rarely used but possible)
            if value <= 8:
                return "s.num_parts <= 300", []  # Kid-friendly sizes
            elif value <= 12:
                return "s.num_parts <= 800", []  # Youth-friendly sizes  
            else:
                return "", []  # No upper constraint for teens/adults
                
        elif constraint_type == ConstraintType.EXCLUDE_OWNED:
            return """NOT EXISTS (
                SELECT 1 FROM user_collections uc 
                WHERE uc.user_id = %s AND uc.set_num = s.set_num
            )""", [value]
            
        elif constraint_type == ConstraintType.EXCLUDE_WISHLISTED:
            return """NOT EXISTS (
                SELECT 1 FROM user_wishlists uw 
                WHERE uw.user_id = %s AND uw.set_num = s.set_num
            )""", [value]
            
        elif constraint_type == ConstraintType.COMPLEXITY_MAX:
            # Map complexity to piece count ranges
            complexity_limits = {
                'simple': 200,
                'moderate': 800,  
                'complex': 999999  # No upper limit for complex
            }
            max_pieces = complexity_limits.get(value, 999999)
            return "s.num_parts <= %s", [max_pieces]
            
        elif constraint_type == ConstraintType.COMPLEXITY_MIN:
            # Map complexity to minimum piece counts
            complexity_minimums = {
                'simple': 0,
                'moderate': 200,
                'complex': 800
            }
            min_pieces = complexity_minimums.get(value, 0)
            return "s.num_parts >= %s", [min_pieces]
            
        elif constraint_type == ConstraintType.AVAILABILITY:
            # For availability, we'll assume sets from recent years are more likely available
            # This is an approximation - real availability would require external data
            current_year = datetime.now().year
            return "s.year >= %s", [current_year - 5]  # Last 5 years
        
        # Default: no constraint
        return "", []
    
    def _get_theme_ids(self, theme_names: List[str]) -> List[int]:
        """
        Get theme IDs from theme names with caching
        
        Args:
            theme_names: List of theme names
            
        Returns:
            List of matching theme IDs
        """
        cache_key = tuple(sorted(theme_names))
        
        if cache_key in self._theme_cache:
            return self._theme_cache[cache_key]
        
        try:
            cursor = self.dbcon.cursor()
            
            # Build query for theme name matching (case-insensitive)
            theme_conditions = []
            params = []
            
            for theme_name in theme_names:
                theme_conditions.append("LOWER(name) LIKE LOWER(%s)")
                params.append(f"%{theme_name}%")
            
            query = f"""
            SELECT id, name FROM themes 
            WHERE {' OR '.join(theme_conditions)}
            """
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            theme_ids = [row[0] for row in results]
            
            # Cache the result
            self._theme_cache[cache_key] = theme_ids
            
            logger.info(f"🎯 Found {len(theme_ids)} theme IDs for themes: {theme_names}")
            
            return theme_ids
            
        except Exception as e:
            logger.error(f"❌ Error getting theme IDs: {e}")
            return []
    
    def _check_constraint_violations(self, 
                                    constraints: List[HardConstraint],
                                    candidate_set_nums: Optional[List[str]] = None) -> List[ConstraintViolation]:
        """
        Check which constraints caused significant filtering and suggest alternatives
        
        Args:
            constraints: Applied constraints
            candidate_set_nums: Original candidate sets
            
        Returns:
            List of constraint violations with suggestions
        """
        violations = []
        
        try:
            # Get total count without constraints
            cursor = self.dbcon.cursor()
            
            if candidate_set_nums:
                cursor.execute(
                    "SELECT COUNT(*) FROM sets WHERE set_num = ANY(%s) AND num_parts > 0",
                    [candidate_set_nums]
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM sets WHERE num_parts > 0")
                
            total_without_constraints = cursor.fetchone()[0]
            
            # Check each constraint individually
            for constraint in constraints:
                # Apply just this constraint
                single_constraint_sql = self._build_constraint_sql([constraint], candidate_set_nums)
                cursor.execute(single_constraint_sql['query'], single_constraint_sql['params'])
                results = cursor.fetchall()
                remaining_count = len(results)
                
                # If constraint eliminated more than 80% of results, it's potentially too restrictive
                elimination_rate = (total_without_constraints - remaining_count) / total_without_constraints
                
                if elimination_rate > 0.8:
                    suggestions = self._generate_constraint_alternatives(constraint)
                    
                    violations.append(ConstraintViolation(
                        constraint=constraint,
                        violating_count=total_without_constraints - remaining_count,
                        total_count=total_without_constraints,
                        message=f"Constraint '{constraint.description}' eliminated {elimination_rate:.1%} of sets",
                        suggested_alternatives=suggestions
                    ))
            
        except Exception as e:
            logger.error(f"❌ Error checking constraint violations: {e}")
        
        return violations
    
    def _generate_constraint_alternatives(self, constraint: HardConstraint) -> List[str]:
        """
        Generate alternative constraint suggestions for overly restrictive constraints
        
        Args:
            constraint: The overly restrictive constraint
            
        Returns:
            List of alternative suggestions
        """
        suggestions = []
        constraint_type = constraint.constraint_type
        value = constraint.value
        
        if constraint_type == ConstraintType.PIECES_MAX:
            suggestions.extend([
                f"Try increasing to {int(value * 1.5)} pieces",
                f"Consider {int(value * 2)} pieces for more options",
                "Remove piece count limit and use price instead"
            ])
            
        elif constraint_type == ConstraintType.PIECES_MIN:
            suggestions.extend([
                f"Try decreasing to {int(value * 0.7)} pieces", 
                f"Consider {int(value * 0.5)} pieces for more options",
                "Remove minimum piece requirement"
            ])
            
        elif constraint_type == ConstraintType.PRICE_MAX:
            suggestions.extend([
                f"Try increasing budget to ${value * 1.3:.2f}",
                f"Consider ${value * 1.5:.2f} for more premium options",
                "Look for sales or discounted sets"
            ])
            
        elif constraint_type == ConstraintType.THEMES_REQUIRED:
            suggestions.extend([
                "Try broader theme categories (e.g., 'space' instead of 'Star Wars')",
                "Consider related themes",
                "Remove theme restriction and browse by interest category"
            ])
            
        elif constraint_type == ConstraintType.AGE_MIN:
            suggestions.extend([
                f"Try age {value - 2}+ for more options",
                "Consider that age ratings are conservative",
                "Look at similar complexity levels across age ranges"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _update_performance_stats(self, constraint_count: int, result_count: int, filter_time: float):
        """Update internal performance statistics"""
        self.performance_stats['total_constraints_applied'] += constraint_count
        self.performance_stats['total_sets_filtered'] += result_count
        
        # Update average filter time
        current_avg = self.performance_stats['average_filter_time_ms']
        total_operations = self.performance_stats.get('operation_count', 0) + 1
        new_avg = ((current_avg * (total_operations - 1)) + filter_time) / total_operations
        
        self.performance_stats['average_filter_time_ms'] = new_avg
        self.performance_stats['operation_count'] = total_operations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics for the constraint filter"""
        return self.performance_stats.copy()
    
    def clear_cache(self):
        """Clear internal caches"""
        self._constraint_cache.clear()
        self._theme_cache.clear()
        self._set_metadata_cache.clear()
        logger.info("🧹 Hard constraint filter caches cleared")

# Utility functions for common constraint patterns

def create_budget_constraints(budget_max: float, budget_min: float = None) -> List[HardConstraint]:
    """Create price-based constraints for budget filtering"""
    constraints = []
    if budget_max:
        constraints.append(HardConstraint(
            ConstraintType.PRICE_MAX,
            budget_max,
            description=f"Budget limit: ${budget_max:.2f}"
        ))
    if budget_min:
        constraints.append(HardConstraint(
            ConstraintType.PRICE_MIN, 
            budget_min,
            description=f"Minimum spend: ${budget_min:.2f}"
        ))
    return constraints

def create_age_appropriate_constraints(age: int, strict: bool = True) -> List[HardConstraint]:
    """Create age-appropriate constraints based on target age"""
    constraints = []
    
    if strict:
        # Strict age constraints
        constraints.append(HardConstraint(
            ConstraintType.AGE_MIN,
            max(4, age - 2),
            description=f"Age appropriate for {age} year old (strict)"
        ))
    else:
        # Flexible age constraints
        constraints.append(HardConstraint(
            ConstraintType.AGE_MIN,
            max(4, age - 4), 
            description=f"Age appropriate for {age} year old (flexible)"
        ))
    
    return constraints

def create_size_constraints(size_category: str) -> List[HardConstraint]:
    """Create piece count constraints based on size category"""
    size_ranges = {
        'mini': (1, 50),
        'small': (51, 200), 
        'medium': (201, 800),
        'large': (801, 2000),
        'xl': (2001, 10000)
    }
    
    if size_category.lower() in size_ranges:
        min_pieces, max_pieces = size_ranges[size_category.lower()]
        return [
            HardConstraint(
                ConstraintType.PIECES_MIN,
                min_pieces,
                description=f"{size_category.title()} size minimum"
            ),
            HardConstraint(
                ConstraintType.PIECES_MAX,
                max_pieces,
                description=f"{size_category.title()} size maximum"
            )
        ]
    
    return []
