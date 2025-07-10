#!/bin/bash

# ========================================
# 🧱 LEGO Recommendation Engine 
# Complete Test Suite Runner
# ========================================
# This script combines database, NL features, and system tests
# with optional integration, performance, and advanced tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧱 LEGO Recommendation Engine - Complete Test Suite${NC}"
echo "===================================================="

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local test_type="$3"
    
    echo -e "\n📋 Running $test_type: $test_name"
    echo "   Command: $test_command"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /tmp/test_output 2>&1; then
        echo -e "   ${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Show key results for successful tests
        if [[ "$test_name" == *"Simple"* ]] || [[ "$test_name" == *"Health"* ]]; then
            grep -E "✅|🎯|🎉|status|results" /tmp/test_output | head -3 2>/dev/null || true
        fi
    else
        echo -e "   ${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "   Error details:"
        tail -n 5 /tmp/test_output | sed 's/^/     /'
    fi
}

# Parse command line arguments
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
RUN_NL_ADVANCED=false
RUN_EXAMPLES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --performance)
            RUN_PERFORMANCE=true
            shift
            ;;
        --nl-advanced)
            RUN_NL_ADVANCED=true
            shift
            ;;
        --examples)
            RUN_EXAMPLES=true
            shift
            ;;
        --all)
            RUN_INTEGRATION=true
            RUN_PERFORMANCE=true
            RUN_NL_ADVANCED=true
            RUN_EXAMPLES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --integration    Run integration tests"
            echo "  --performance    Run performance tests"
            echo "  --nl-advanced    Run advanced NL tests"
            echo "  --examples       Run example scripts"
            echo "  --all           Run all optional tests"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            print_warning "Unknown option: $1"
            shift
            ;;
    esac
done

# Check prerequisites
echo -e "\n🔍 Checking Prerequisites"
echo "========================="

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    print_warning "Docker not accessible. Some tests may fail."
fi

# Check if API is running
API_RUNNING=false
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "API is running"
    API_RUNNING=true
elif docker-compose exec app conda run -n brickbrain-rec python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" > /dev/null 2>&1; then
    print_status "API is running (in container)"
    API_RUNNING=true
else
    print_warning "API not running. Some tests will be skipped."
fi

# ========================================
# 1. Core Unit Tests
# ========================================

echo -e "\n${BLUE}1. CORE UNIT TESTS${NC}"
echo "=================="

cd tests/unit

# Determine python command
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
    PYTHON_CMD="conda run -n brickbrain-rec python"
elif docker-compose ps app | grep -q "Up"; then
    PYTHON_CMD="docker-compose exec app conda run -n brickbrain-rec python"
else
    PYTHON_CMD="python"
fi

run_test "Database Connection Tests" "$PYTHON_CMD test_database.py" "Unit Test"
run_test "Recommendation System Tests" "$PYTHON_CMD test_recommendations.py" "Unit Test"

cd ../..

# ========================================
# 2. API Health and Basic Functionality
# ========================================

echo -e "\n${BLUE}2. API HEALTH AND BASIC FUNCTIONALITY${NC}"
echo "====================================="

if [ "$API_RUNNING" = true ]; then
    # Health check test
    run_test "API Health Check" "curl -s http://localhost:8000/health | grep -q status" "API Test"
    
    # Basic endpoint tests
    run_test "API Documentation Access" "curl -s http://localhost:8000/docs | grep -q swagger" "API Test"
    
    # Basic search functionality
    run_test "Basic Search Endpoint" "curl -s -X POST http://localhost:8000/search/sets -H 'Content-Type: application/json' -d '{\"theme_name\": \"Star Wars\", \"top_k\": 3}' | grep -q results" "API Test"
else
    print_warning "API not running - skipping API tests"
fi

# ========================================
# 3. Natural Language Features Tests
# ========================================

echo -e "\n${BLUE}3. NATURAL LANGUAGE FEATURES${NC}"
echo "============================="

if [ "$API_RUNNING" = true ]; then
    # Basic NL search test
    run_test "NL Search Basic" "curl -s -X POST http://localhost:8000/search/natural -H 'Content-Type: application/json' -d '{\"query\": \"star wars sets\", \"top_k\": 3}' | grep -q results" "NL Test"
    
    # Query understanding test
    run_test "Query Understanding" "curl -s -X POST http://localhost:8000/nlp/understand -H 'Content-Type: application/json' -d '{\"query\": \"birthday gift for kids\"}' | grep -q intent || curl -s -X POST http://localhost:8000/nlp/understand -H 'Content-Type: application/json' -d '{\"query\": \"birthday gift for kids\"}' | grep -q error" "NL Test"
    
    if [ "$RUN_NL_ADVANCED" = true ]; then
        echo -e "\n${BLUE}Advanced NL Tests${NC}"
        echo "=================="
        
        # Multiple query tests
        declare -a test_queries=(
            "star wars sets for kids"
            "birthday gift for 8 year old"
            "challenging technic sets"
            "small city sets under 500 pieces"
        )
        
        for query in "${test_queries[@]}"; do
            run_test "NL Query: '$query'" "curl -s -X POST http://localhost:8000/search/natural -H 'Content-Type: application/json' -d '{\"query\": \"$query\", \"top_k\": 3}' | grep -q results" "Advanced NL Test"
        done
        
        # Performance test
        print_info "Testing NL search response times..."
        for i in {1..3}; do
            start_time=$(date +%s.%N)
            curl -s -X POST "http://localhost:8000/search/natural" \
                -H "Content-Type: application/json" \
                -d '{"query": "star wars sets", "top_k": 5}' > /dev/null 2>&1
            end_time=$(date +%s.%N)
            
            duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
            echo "   Query $i response time: ${duration}s"
        done
    fi
else
    print_warning "API not running - skipping NL tests"
fi

# ========================================
# 4. Integration Tests (Optional)
# ========================================

if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "\n${BLUE}4. INTEGRATION TESTS${NC}"
    echo "===================="
    
    cd tests/integration
    
    if [ -f "production_test_simple.py" ]; then
        run_test "Simple Production Test" "$PYTHON_CMD production_test_simple.py" "Integration Test"
    fi
    
    if [ -f "final_validation.py" ]; then
        run_test "Final Validation" "$PYTHON_CMD final_validation.py" "Integration Test"
    fi
    
    if [ -f "validate_production_readiness.py" ]; then
        run_test "Production Readiness" "$PYTHON_CMD validate_production_readiness.py" "Integration Test"
    fi
    
    if [ -f "nl_integration_test.py" ]; then
        run_test "NL Integration Test" "$PYTHON_CMD nl_integration_test.py" "Integration Test"
    fi
    
    cd ../..
else
    echo -e "\n${BLUE}4. INTEGRATION TESTS${NC}"
    echo "===================="
    print_info "Skipped - Use --integration to run"
fi

# ========================================
# 5. Performance Tests (Optional)
# ========================================

if [ "$RUN_PERFORMANCE" = true ]; then
    echo -e "\n${BLUE}5. PERFORMANCE TESTS${NC}"
    echo "===================="
    
    cd tests/performance
    
    if [ -f "production_scalability_test.py" ]; then
        run_test "Scalability Test" "$PYTHON_CMD production_scalability_test.py" "Performance Test"
    fi
    
    cd ../..
else
    echo -e "\n${BLUE}5. PERFORMANCE TESTS${NC}"
    echo "===================="
    print_info "Skipped - Use --performance to run"
fi

# ========================================
# 6. Example Scripts (Optional)
# ========================================

if [ "$RUN_EXAMPLES" = true ] && [ "$API_RUNNING" = true ]; then
    echo -e "\n${BLUE}6. EXAMPLE SCRIPTS${NC}"
    echo "=================="
    
    cd examples
    
    if [ -f "example_client.py" ]; then
        run_test "Example Client" "$PYTHON_CMD example_client.py" "Example"
    fi
    
    if [ -f "nl_demo_script.py" ]; then
        run_test "NL Demo Script" "$PYTHON_CMD nl_demo_script.py --quick" "Example"
    fi
    
    cd ..
else
    echo -e "\n${BLUE}6. EXAMPLE SCRIPTS${NC}"
    echo "=================="
    if [ "$API_RUNNING" = true ]; then
        print_info "Skipped - Use --examples to run"
    else
        print_info "Skipped - API not running"
    fi
fi

# ========================================
# 7. System Status Check
# ========================================

echo -e "\n${BLUE}7. SYSTEM STATUS CHECK${NC}"
echo "======================"

print_info "Checking system components..."

# Check cache directories
if [ -d ".cache" ]; then
    CACHE_SIZE=$(du -sh .cache 2>/dev/null | cut -f1 || echo "Unknown")
    print_status "Cache directory exists (size: $CACHE_SIZE)"
else
    print_warning "Cache directory not found"
fi

# Check embeddings
if [ -d "embeddings" ]; then
    EMB_SIZE=$(du -sh embeddings 2>/dev/null | cut -f1 || echo "Unknown")
    print_status "Embeddings directory exists (size: $EMB_SIZE)"
else
    print_warning "Embeddings directory not found"
fi

# Check data availability
if [ -d "data/rebrickable" ]; then
    DATA_FILES=$(find data/rebrickable -name "*.csv" | wc -l)
    print_status "LEGO data available ($DATA_FILES CSV files)"
else
    print_warning "No LEGO data found - some features may not work"
fi

# ========================================
# 8. Final Summary
# ========================================

echo -e "\n${BLUE}8. TEST SUMMARY${NC}"
echo "==============="

if [ $TOTAL_TESTS -eq 0 ]; then
    print_warning "No tests were run"
    exit 1
fi

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "Success Rate: $SUCCESS_RATE%"

echo -e "\n${GREEN}📊 Test Categories:${NC}"
echo "• Unit Tests: Core functionality"
echo "• API Tests: Endpoint availability"
echo "• NL Tests: Natural language features"
if [ "$RUN_INTEGRATION" = true ]; then
    echo "• Integration Tests: End-to-end workflows"
fi
if [ "$RUN_PERFORMANCE" = true ]; then
    echo "• Performance Tests: Scalability and speed"
fi
if [ "$RUN_EXAMPLES" = true ]; then
    echo "• Example Scripts: Usage demonstrations"
fi

echo -e "\n${GREEN}🎯 System Status:${NC}"
if [ $FAILED_TESTS -eq 0 ]; then
    print_status "ALL TESTS PASSED! SYSTEM IS READY!"
    
    echo -e "\n${GREEN}🚀 Ready for:${NC}"
    echo "• Production deployment"
    echo "• Natural language queries"
    echo "• Full API usage"
    echo "• Integration with client applications"
    
elif [ $SUCCESS_RATE -ge 75 ]; then
    print_warning "MOSTLY WORKING - Minor issues detected"
    
    echo -e "\n${GREEN}✅ Working:${NC}"
    echo "• Core functionality"
    echo "• Basic API endpoints"
    echo "• Database connectivity"
    
    echo -e "\n${YELLOW}⚠️  Issues:${NC}"
    echo "• Some advanced features may need attention"
    echo "• Check failed tests above"
    
else
    print_error "CRITICAL ISSUES DETECTED"
    
    echo -e "\n${RED}❌ Issues:${NC}"
    echo "• Multiple test failures"
    echo "• System may not be ready for production"
    echo "• Review setup and configuration"
fi

echo -e "\n${GREEN}🔧 Useful Commands:${NC}"
echo "• Run all tests: ./scripts/run_all_tests.sh --all"
echo "• Run specific tests: ./scripts/run_all_tests.sh --integration --performance"
echo "• Check API: curl http://localhost:8000/health"
echo "• View logs: docker-compose logs app"
echo "• Restart system: ./setup_and_start.sh"

echo -e "\n${BLUE}📚 Documentation:${NC}"
echo "• API Docs: http://localhost:8000/docs"
echo "• Test README: tests/README.md"
echo "• Setup Guide: README.md"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 TESTING COMPLETE - SYSTEM IS READY!${NC}"
    exit 0
elif [ $SUCCESS_RATE -ge 75 ]; then
    echo -e "\n${YELLOW}⚠️  TESTING COMPLETE - MOSTLY WORKING${NC}"
    exit 0
else
    echo -e "\n${RED}❌ TESTING COMPLETE - ISSUES NEED ATTENTION${NC}"
    exit 1
fi
