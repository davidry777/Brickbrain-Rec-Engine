# 🧱 LEGO Recommendation Engine - Test Suite

This directory contains comprehensive tests for the LEGO Recommendation Engine, including database connectivity, API functionality, natural language processing, and system integration tests.

## 🚀 Quick Start

### Run All Tests (Basic)
```bash
./run_all_tests.sh
```

### Run All Tests (Including Optional)
```bash
./run_all_tests.sh --all
```

### Run Specific Test Categories
```bash
# Integration tests only
./run_all_tests.sh --integration

# Performance tests only
./run_all_tests.sh --performance

# Advanced NL tests only
./run_all_tests.sh --nl-advanced

# Example scripts
./run_all_tests.sh --examples

# Combine multiple categories
./run_all_tests.sh --integration --performance
```

## 📁 Test Structure

### Core Tests (Always Run)
- **Unit Tests** (`unit/`): Core functionality testing
  - `test_database.py`: Database connection and queries
  - `test_recommendations.py`: Recommendation system logic
- **API Tests**: Basic endpoint availability and health checks
- **NL Basic Tests**: Essential natural language functionality

### Optional Tests (Run with Flags)
- **Integration Tests** (`integration/`): End-to-end workflows
- **Performance Tests** (`performance/`): Speed and scalability
- **Advanced NL Tests**: Comprehensive natural language testing
- **Example Scripts** (`examples/`): Usage demonstrations

## 🧪 Test Categories

### 1. Unit Tests (`unit/`)

Tests core functionality without external dependencies.

#### Database Tests (`test_database.py`)
- Database connection
- Schema validation
- Basic CRUD operations
- Data integrity checks

#### Recommendation Tests (`test_recommendations.py`)
- Recommendation algorithm logic
- Filtering functionality
- Result ranking
- Edge case handling

**Run individually:**
```bash
cd tests/unit
python test_database.py
python test_recommendations.py
```

### 2. Integration Tests (`integration/`)

Tests complete workflows and system interactions.

#### Core Integration Tests
- `production_test_simple.py`: Basic production readiness
- `final_validation.py`: Comprehensive system validation
- `validate_production_readiness.py`: Deployment readiness check

#### Natural Language Integration (`nl_integration_test.py`)
- Database + NL components integration
- API + NL features integration
- End-to-end NL query processing
- Performance benchmarking
- Error handling validation

**Features tested:**
- NL search API endpoints
- Query understanding
- Response time performance
- Error handling
- Data availability validation

**Run individually:**
```bash
cd tests/integration
python nl_integration_test.py
python production_test_simple.py
```

### 3. Performance Tests (`performance/`)

Tests system performance under various conditions.

#### Scalability Test (`production_scalability_test.py`)
- Concurrent request handling
- Memory usage monitoring
- Response time under load
- Database performance
- API throughput testing

**Run individually:**
```bash
cd tests/performance
python production_scalability_test.py
```

## 🎯 Test Execution Options

### Basic Testing (Default)
Runs essential tests that validate core functionality:
- Unit tests
- API health checks
- Basic NL functionality
- System status verification

### Extended Testing (Optional)
Additional tests that provide comprehensive validation:
- **Integration Tests** (`--integration`): End-to-end workflows
- **Performance Tests** (`--performance`): Load and speed testing
- **Advanced NL Tests** (`--nl-advanced`): Comprehensive NL validation
- **Example Scripts** (`--examples`): Usage demonstrations

## 🔧 Test Configuration

### Environment Variables
Tests use the same environment variables as the application:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=brickbrain
DB_USER=brickbrain
DB_PASSWORD=brickbrain_password
```

### Prerequisites
1. **Database**: PostgreSQL running with test data
2. **API Server**: FastAPI server running on localhost:8000
3. **Python Environment**: Conda environment with all dependencies
4. **Docker**: For containerized testing (optional)

### Setup for Testing
```bash
# Complete setup and start
./setup_and_start.sh

# Then run tests
./run_all_tests.sh --all
```

## 📊 Test Output

### Success Indicators
- ✅ **Green checkmarks**: Test passed
- 📋 **Test name**: What is being tested
- 🎯 **Summary stats**: Pass/fail counts and success rate

### Warning Indicators
- ⚠️ **Yellow warnings**: Non-critical issues
- 🔄 **Retries**: Automatic retry attempts
- 📝 **Partial results**: Some features working

### Error Indicators
- ❌ **Red X marks**: Test failed
- 🚨 **Critical errors**: System-breaking issues
- 💥 **Exceptions**: Unexpected failures

### Example Output
```
🧱 LEGO Recommendation Engine - Complete Test Suite
====================================================

🔍 Checking Prerequisites
=========================
✅ API is running
✅ Docker is accessible

1. CORE UNIT TESTS
==================
📋 Running Unit Test: Database Connection Tests
   ✅ PASSED

📋 Running Unit Test: Recommendation System Tests  
   ✅ PASSED

2. API HEALTH AND BASIC FUNCTIONALITY
=====================================
📋 Running API Test: API Health Check
   ✅ PASSED

📋 Running API Test: Basic Search Endpoint
   ✅ PASSED

📋 Running NL Test: NL Search Basic
   ✅ PASSED

8. TEST SUMMARY
===============
Total Tests: 12
Passed: 12
Failed: 0
Success Rate: 100%

🎉 ALL TESTS PASSED! SYSTEM IS READY!
```

## 🐛 Troubleshooting

### Common Issues

#### API Not Running
```
⚠️ API not running. Some tests will be skipped.
```
**Solution:**
```bash
./setup_and_start.sh
```

#### Database Connection Failed
```
❌ Database connection failed: connection refused
```
**Solution:**
```bash
docker-compose up -d postgres
# Wait 30 seconds, then retry
```

#### No Test Data
```
⚠️ No LEGO data found - some features may not work
```
**Solution:**
```bash
# Ensure data is loaded
docker-compose exec app conda run -n brickbrain-rec python src/scripts/upload_rebrickable_data.py
```

#### Docker Issues
```
⚠️ Docker not accessible. Some tests may fail.
```
**Solution:**
```bash
# Start Docker Desktop, then:
docker-compose down
docker-compose up -d
```

### Test-Specific Issues

#### Integration Tests Failing
- Check API server is running: `curl http://localhost:8000/health`
- Verify database connectivity: `docker-compose exec postgres pg_isready`
- Ensure all dependencies installed: `docker-compose exec app conda list`

#### Performance Tests Slow
- Monitor system resources: `docker stats`
- Check database performance: Review query execution times
- Consider data volume: Large datasets may impact performance

#### NL Tests Not Working
- Verify NL dependencies: `docker-compose exec app conda list | grep -E "(nltk|spacy|sklearn)"`
- Check model downloads: Ensure NLTK and spaCy models are available
- Validate embeddings: Verify sentence-transformers model is loaded

## 🔄 Continuous Integration

### Automated Testing
For CI/CD pipelines:
```bash
# Basic validation (fast)
./run_all_tests.sh

# Full validation (comprehensive)
./run_all_tests.sh --all
```

### Exit Codes
- `0`: All tests passed
- `1`: Some tests failed (check logs)

### Integration with CI
```yaml
# Example GitHub Actions step
- name: Run Tests
  run: |
    ./setup_and_start.sh
    ./run_all_tests.sh --integration --performance
```

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Setup Guide**: ../README.md
- **Natural Language Features**: ../NL_FEATURES_README.md
- **API Guide**: ../API_README.md

## 🎯 Best Practices

1. **Run basic tests first**: Validate core functionality before advanced features
2. **Check prerequisites**: Ensure all services are running before testing
3. **Monitor resources**: Performance tests can be resource-intensive
4. **Review logs**: Check `docker-compose logs app` for detailed error information
5. **Incremental testing**: Run specific test categories when developing new features
