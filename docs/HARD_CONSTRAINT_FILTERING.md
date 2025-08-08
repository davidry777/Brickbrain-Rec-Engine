# 🔒 Hard Constraint Filtering System

## Overview

The Hard Constraint Filtering System is a powerful new feature for the LEGO Brickbrain Recommendation Engine that enforces **absolute, non-negotiable requirements** on recommendations. Unlike soft filtering that influences scoring, hard constraints create strict barriers that **completely exclude** results that don't meet the specified criteria.

## 🎯 Key Features

### ✅ Absolute Enforcement
- **No fuzzy matching** - constraints are binary (pass/fail)
- **Zero tolerance** - violating even one constraint eliminates a result
- **Clear boundaries** - users get exactly what they ask for

### 🔧 Comprehensive Constraint Types
- **💰 Price constraints**: Budget limits with realistic price estimation
- **🔢 Piece count constraints**: Minimum/maximum build complexity  
- **👶 Age constraints**: Age-appropriate content filtering
- **📅 Year constraints**: Release date filtering (vintage vs modern)
- **🎨 Theme constraints**: Required/excluded LEGO themes
- **🧩 Complexity constraints**: Build difficulty levels
- **👤 User-specific constraints**: Exclude owned/wishlisted sets
- **📦 Availability constraints**: Currently purchasable sets only

### 🚀 Performance Optimized
- **SQL-based filtering** at database level for maximum speed
- **Intelligent caching** for repeated theme/constraint lookups
- **Sub-100ms response times** for most constraint combinations
- **Batch processing** for multiple constraints simultaneously

### 🧠 Smart Violation Detection
- **Automatic detection** of overly restrictive constraints
- **Elimination rate analysis** (warns when >80% of sets are filtered out)
- **Alternative suggestions** when constraints can't be met
- **Clear feedback** on which constraints are causing issues

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Request   │    │  Constraint      │    │  Recommendation │
│  with Constraints├──→│  Filter System   ├──→│     Engine      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   PostgreSQL     │
                       │   Database       │
                       │  (Optimized SQL) │
                       └──────────────────┘
```

### Core Components

1. **HardConstraintFilter**: Main filtering engine
2. **HardConstraint**: Individual constraint definitions  
3. **ConstraintResult**: Filter application results with stats
4. **Enhanced API Endpoints**: New constraint-aware endpoints
5. **Integration Layer**: Seamless integration with existing recommenders

## 📋 Usage Examples

### Basic Constraint Creation

```python
from hard_constraint_filter import HardConstraintFilter, ConstraintType, HardConstraint

# Create constraint filter
constraint_filter = HardConstraintFilter(db_connection)

# Create budget constraint
budget_constraint = HardConstraint(
    ConstraintType.PRICE_MAX, 
    100.0,
    description="Must cost less than $100"
)

# Apply constraint
result = constraint_filter.apply_constraints([budget_constraint])
valid_sets = result.valid_set_nums  # Only sets under $100
```

### Multiple Constraints

```python
# Create comprehensive constraint set
constraints = constraint_filter.create_constraint_set(
    price_max=150.0,           # Budget limit
    pieces_min=200,            # Substantial build
    pieces_max=1500,           # Not overwhelming  
    age_min=8,                 # Age appropriate
    required_themes=["City", "Creator"],  # Specific themes
    excluded_themes=["Duplo"], # Exclude inappropriate
    max_complexity="moderate", # Manageable difficulty
    year_min=2018             # Modern sets only
)

result = constraint_filter.apply_constraints(constraints)
```

### API Integration

```python
# Enhanced recommendation request
request = EnhancedRecommendationRequest(
    user_id=123,
    top_k=10,
    price_max=100.0,
    pieces_min=300,
    required_themes=["Star Wars"],
    exclude_owned=True
)

# Get constrained recommendations
response = await get_constrained_recommendations(request)
```

### Natural Language Integration

```python
# Natural language with automatic constraint extraction
query = "Star Wars sets under $75 for my 10-year-old with 500+ pieces"

# System automatically extracts:
# - required_themes: ["Star Wars"]  
# - price_max: 75.0
# - age_min: 8, age_max: 12
# - pieces_min: 500

response = await natural_language_search_constrained(query)
```

## 🛠️ API Endpoints

### POST `/recommendations/constrained`
Enhanced recommendation endpoint with full constraint support.

**Request Body:**
```json
{
  "user_id": 123,
  "top_k": 10,
  "price_max": 100.0,
  "pieces_min": 200,
  "pieces_max": 1000,
  "age_min": 8,
  "required_themes": ["City", "Creator"],
  "excluded_themes": ["Duplo"],
  "max_complexity": "moderate",
  "exclude_owned": true
}
```

**Response:**
```json
{
  "recommendations": [...],
  "constraint_summary": {
    "total_constraints_applied": 7,
    "valid_sets_found": 234,
    "recommendations_returned": 10,
    "filtering_effective": true
  },
  "violations": [],
  "performance_stats": {
    "filter_time_ms": 45.2
  }
}
```

### POST `/search/natural/constrained`
Natural language search with automatic constraint extraction.

**Request Body:**
```json
{
  "query": "LEGO sets under $50 for my 8-year-old nephew who loves cars",
  "top_k": 10
}
```

**Automatically Extracts:**
- `price_max: 50.0`
- `age_min: 6, age_max: 10` (nephew context)
- `required_themes: ["City", "Speed Champions"]` (cars context)

## 🎨 Constraint Types Reference

### Price Constraints
```python
# Budget limits (estimated from piece count)
ConstraintType.PRICE_MAX    # Maximum price
ConstraintType.PRICE_MIN    # Minimum price
```

### Physical Constraints  
```python
# Piece count limits
ConstraintType.PIECES_MAX   # Maximum pieces
ConstraintType.PIECES_MIN   # Minimum pieces

# Release year limits
ConstraintType.YEAR_MAX     # Latest release year
ConstraintType.YEAR_MIN     # Earliest release year
```

### Content Constraints
```python
# Theme filtering
ConstraintType.THEMES_REQUIRED  # Must be from these themes
ConstraintType.THEMES_EXCLUDED  # Must NOT be from these themes

# Age appropriateness
ConstraintType.AGE_MIN         # Minimum age suitability
ConstraintType.AGE_MAX         # Maximum age suitability

# Build complexity
ConstraintType.COMPLEXITY_MAX  # Maximum difficulty
ConstraintType.COMPLEXITY_MIN  # Minimum difficulty
```

### User-Specific Constraints
```python
# Personal collection filters
ConstraintType.EXCLUDE_OWNED      # Exclude user's owned sets
ConstraintType.EXCLUDE_WISHLISTED # Exclude wishlisted sets

# Availability
ConstraintType.AVAILABILITY       # Must be currently available
```

## 🧪 Testing & Validation

### Run Tests
```bash
# Run comprehensive test suite
python tests/unit/test_hard_constraint_filter.py

# Run interactive demo
python examples/hard_constraint_demo.py
```

### Test Scenarios Covered
- ✅ Constraint creation and validation
- ✅ SQL query generation  
- ✅ Multiple constraint combinations
- ✅ Overly restrictive constraint handling
- ✅ Performance monitoring
- ✅ Cache functionality
- ✅ API integration
- ✅ Natural language integration

## 📊 Performance Characteristics

### Typical Performance
- **Simple constraints** (1-2 filters): ~15-30ms
- **Multiple constraints** (3-5 filters): ~30-60ms  
- **Complex constraints** (6+ filters): ~60-100ms
- **Theme lookups**: Cached after first use
- **Memory usage**: Minimal (stateless processing)

### Scalability
- **Database-optimized**: Uses indexed SQL queries
- **Concurrent-safe**: No shared state between requests
- **Cache-efficient**: Theme and metadata caching
- **Resource-friendly**: Low memory footprint

## 🚨 Error Handling

### Constraint Violations
When constraints are too restrictive:
```json
{
  "recommendations": [],
  "violations": [{
    "constraint_description": "Maximum $5 budget",
    "violating_count": 24891,
    "total_count": 25000,
    "elimination_rate": 0.996,
    "suggested_alternatives": [
      "Try increasing budget to $15",
      "Consider $25 for more premium options"
    ]
  }]
}
```

### System Responses
- **Empty results**: Clear explanation of constraint conflicts
- **Performance issues**: Automatic optimization suggestions
- **Invalid constraints**: Detailed validation errors
- **Database issues**: Graceful degradation with fallbacks

## 🔮 Future Enhancements

### Planned Features
- **Soft constraint scoring**: Influence recommendation ranking
- **Constraint templates**: Pre-built constraint sets for common scenarios
- **Machine learning**: Learn optimal constraints from user behavior
- **Advanced pricing**: Real-time price data integration
- **Availability API**: Live inventory checking
- **Constraint explanations**: Natural language constraint descriptions

### Integration Roadmap
- **Mobile API**: Optimized endpoints for mobile apps
- **Real-time notifications**: Alert when constrained sets become available
- **Batch processing**: Bulk constraint application for large datasets
- **Analytics**: Constraint usage analytics and optimization

## 💡 Best Practices

### Effective Constraint Design
1. **Start broad, narrow down**: Begin with loose constraints, tighten as needed
2. **Combine related constraints**: Group logical constraints together
3. **Monitor violations**: Pay attention to elimination rates
4. **Use constraint suggestions**: System provides smart alternatives
5. **Test edge cases**: Validate with extreme constraint combinations

### Performance Optimization
1. **Cache theme lookups**: Reuse theme ID mappings
2. **Batch constraints**: Apply multiple constraints in single query
3. **Monitor timing**: Track filter performance over time
4. **Database tuning**: Ensure proper indexing on filtered columns

### User Experience
1. **Clear feedback**: Explain why constraints eliminated results
2. **Alternative suggestions**: Offer viable constraint modifications
3. **Progressive disclosure**: Start simple, reveal advanced options
4. **Context awareness**: Use user history to suggest appropriate constraints

## 🤝 Contributing

### Adding New Constraint Types
1. Define new `ConstraintType` enum value
2. Implement SQL generation in `_constraint_to_sql()`
3. Add validation logic in `create_constraint_set()`
4. Create test cases in test suite
5. Update documentation and examples

### Performance Improvements
1. Profile constraint application with various combinations
2. Optimize SQL query generation
3. Enhance caching strategies
4. Add database query optimization

## 📚 Related Documentation

- [API Documentation](../docs/GRADIO_README.md)
- [NLP Integration](../docs/ENHANCED_THEME_DETECTION.md)  
- [Database Schema](../src/db/rebrickable_schema.sql)
- [Testing Guide](../tests/README.md)

---

**Hard Constraint Filtering System** - Ensuring every recommendation meets your exact requirements, every time. 🎯
