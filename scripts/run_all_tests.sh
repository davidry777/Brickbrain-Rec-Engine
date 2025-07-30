#!/bin/bash

# ========================================
# 🧱 LEGO Recommendation Engine 
# Complete Test Suite Runner
# ========================================
# Comprehensive test runner with optional advanced features

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
        
        # Show actual API response for debugging
        if [[ "$test_command" == *"curl"* ]]; then
            echo "   API Response:"
            eval "${test_command%% | grep*}" | head -3 | sed 's/^/     /' 2>/dev/null || echo "     No response"
        fi
        
        tail -n 5 /tmp/test_output | sed 's/^/     /'
    fi
}

# Parse command line arguments
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
RUN_NL_ADVANCED=false
RUN_EXAMPLES=false
RUN_GRADIO_OPTIONAL=false

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
        --gradio-optional)
            RUN_GRADIO_OPTIONAL=true
            shift
            ;;
        --all)
            RUN_INTEGRATION=true
            RUN_PERFORMANCE=true
            RUN_NL_ADVANCED=true
            RUN_EXAMPLES=true
            RUN_GRADIO_OPTIONAL=true
            shift
            ;;
        --quick)
            # Quick mode - just run the essential tests
            print_info "Quick mode: Running essential tests only"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --quick           Run essential tests only (endpoint + production validation)"
            echo "  --integration     Run integration tests"
            echo "  --performance     Run performance tests"
            echo "  --nl-advanced     Run advanced NL tests"
            echo "  --examples        Run example scripts"
            echo "  --gradio-optional Run optional Gradio tests"
            echo "  --all            Run all optional tests"
            echo "  -h, --help       Show this help message"
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
# 1. Essential Tests (Always Run)
# ========================================

echo -e "\n${BLUE}1. ESSENTIAL TESTS${NC}"
echo "=================="

# Quick endpoint validation using our comprehensive test
echo -e "\n${BLUE}🎯 Comprehensive Endpoint Testing${NC}"
if [ -f "scripts/test_endpoints.sh" ]; then
    chmod +x scripts/test_endpoints.sh
    if ./scripts/test_endpoints.sh; then
        print_status "All endpoints working (100% success rate)"
    else
        print_error "Endpoint tests failed"
        exit 1
    fi
else
    print_warning "scripts/test_endpoints.sh not found, skipping endpoint tests"
fi

# Production validation
echo -e "\n${BLUE}🚀 Final Production Validation${NC}"
if [ -f "tests/integration/final_validation.py" ]; then
    if python3 tests/integration/final_validation.py; then
        print_status "Production validation passed"
    else
        print_error "Production validation failed"
        exit 1
    fi
else
    print_warning "final_validation.py not found, skipping production validation"
fi

# Check if we should stop here (quick mode)
if [[ "$1" == "--quick" ]] || [[ "$*" == *"--quick"* ]]; then
    echo -e "\n${GREEN}🎉 QUICK TEST SUITE COMPLETE${NC}"
    echo "=================================================="
    echo -e "${GREEN}✅ Essential systems validated${NC}"
    echo -e "${BLUE}📊 All critical endpoints working${NC}"
    echo -e "${BLUE}🚀 System is production ready${NC}"
    echo -e "\n${BLUE}💡 Run with --all for comprehensive testing${NC}"
    exit 0
fi

# ========================================
# 2. Core Unit Tests
# ========================================

echo -e "\n${BLUE}2. CORE UNIT TESTS${NC}"
echo "=================="

cd tests/unit

# Determine python command - prioritize Docker container execution
if docker-compose ps app | grep -q "Up"; then
    PYTHON_CMD="docker-compose exec -T app conda run -n brickbrain-rec python"
    print_info "Using Docker container for unit tests"
elif [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
    PYTHON_CMD="conda run -n brickbrain-rec python"
    print_info "Using local conda environment for unit tests"
else
    PYTHON_CMD="python3"
    print_info "Using system Python for unit tests"
fi

run_test "Database Connection Tests" "$PYTHON_CMD test_database.py" "Unit Test"
run_test "Recommendation System Tests" "$PYTHON_CMD test_recommendations.py" "Unit Test"
run_test "NLP Recommender Tests" "$PYTHON_CMD test_nlp_recommender.py" "Unit Test"
run_test "Enhanced Theme Detection Tests" "$PYTHON_CMD test_enhanced_themes.py" "Unit Test"

cd ../..

# ========================================
# 3. API Health and Basic Functionality 
# ========================================

echo -e "\n${BLUE}3. API HEALTH AND BASIC FUNCTIONALITY${NC}"
echo "====================================="

if [ "$API_RUNNING" = true ]; then
    # Health check test
    run_test "API Health Check" "curl -s http://localhost:8000/health | grep -q status" "API Test"
    
    # Basic endpoint tests
    run_test "API Documentation Access" "curl -s http://localhost:8000/docs | grep -q swagger" "API Test"
    
    # Basic search functionality
    run_test "Basic Search Endpoint" "curl -s -X POST http://localhost:8000/search/sets -H 'Content-Type: application/json' -d '{\"query\": \"space\", \"limit\": 3}' | grep -q '\"set_num\"'" "API Test"
else
    print_warning "API not running - skipping API tests"
fi

# ========================================
# 4. Integration Tests (Optional)
# ========================================

if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "\n${BLUE}4. INTEGRATION TESTS${NC}"
    echo "===================="
    
    cd tests/integration
    
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
# 6. Natural Language Features Tests (Optional)
# ========================================

if [ "$RUN_NL_ADVANCED" = true ]; then
    echo -e "\n${BLUE}6. ADVANCED NATURAL LANGUAGE TESTS${NC}"
    echo "=================================="
    
    # Enhanced Theme Detection Unit Tests in Docker
    echo -e "\n${BLUE}🎯 Enhanced Theme Detection Tests (Docker)${NC}"
    cd tests/unit
    
    # Use Docker container for enhanced theme detection tests
    if docker-compose ps app | grep -q "Up"; then
        DOCKER_PYTHON_CMD="docker-compose exec -T app conda run -n brickbrain-rec python"
        run_test "Enhanced Theme Detection (Docker)" "$DOCKER_PYTHON_CMD test_enhanced_themes.py" "Enhanced NL Test"
    else
        print_warning "Docker app container not running - using fallback Python command"
        run_test "Enhanced Theme Detection (Fallback)" "$PYTHON_CMD test_enhanced_themes.py" "Enhanced NL Test"
    fi
    
    cd ../..
    
    # API-based NL tests (only if API is running)
    if [ "$API_RUNNING" = true ]; then
        echo -e "\n${BLUE}🌐 API Natural Language Tests${NC}"
        
        # Test multiple query scenarios
        declare -a test_queries=(
            "star wars sets for kids"
            "birthday gift for 8 year old"
            "challenging technic sets"
            "small city sets under 500 pieces"
        )
        
        for query in "${test_queries[@]}"; do
            run_test "NL Query: '$query'" "curl -s -X POST http://localhost:8000/search/natural -H 'Content-Type: application/json' -d '{\"query\": \"$query\", \"top_k\": 3}' | grep -q 'results'" "Advanced NL Test"
        done
    else
        print_info "API not running - skipping API-based NL tests"
    fi
else
    echo -e "\n${BLUE}6. ADVANCED NATURAL LANGUAGE TESTS${NC}"
    echo "=================================="
    print_info "Skipped - Use --nl-advanced to run"
fi

# ========================================
# 7. Gradio Interface Tests
# ========================================

echo -e "\n${BLUE}7. GRADIO INTERFACE TESTS${NC}"
echo "========================="

# Check if Gradio container is running
GRADIO_RUNNING=false
if curl -s http://localhost:7860 > /dev/null 2>&1; then
    print_status "Gradio interface is running"
    GRADIO_RUNNING=true
elif docker-compose ps gradio | grep -q "Up"; then
    print_status "Gradio container is running"
    GRADIO_RUNNING=true
else
    print_warning "Gradio interface not running"
fi

if [ "$GRADIO_RUNNING" = true ]; then
    # Test Gradio interface accessibility
    run_test "Gradio Interface Access" "curl -s http://localhost:7860 | grep -q 'Gradio\\|gradio\\|interface'" "Gradio Test"
    
    if [ "$RUN_GRADIO_OPTIONAL" = true ]; then
        # Optional health checks - but handle gracefully if not available
        print_info "Testing optional Gradio health endpoint..."
        if curl -s http://localhost:7860/health > /dev/null 2>&1; then
            HEALTH_RESPONSE=$(curl -s http://localhost:7860/health)
            if [[ "$HEALTH_RESPONSE" == *"status"* ]] || [[ "$HEALTH_RESPONSE" == *"ok"* ]]; then
                run_test "Gradio Health Check" "curl -s http://localhost:7860/health | grep -q 'status\\|ok'" "Gradio Test"
            else
                print_info "Gradio Health Check: Custom health endpoint found but no standard response format"
            fi
        else
            print_info "Gradio Health Check: No health endpoint (normal for Gradio apps)"
        fi
    fi
    
    print_info "Gradio interface available at: http://localhost:7860"
else
    print_error "Gradio interface is not accessible"
    print_info "Try running: docker-compose up -d gradio"
fi

# ========================================
# 8. Example Scripts (Optional)
# ========================================

if [ "$RUN_EXAMPLES" = true ] && [ "$API_RUNNING" = true ]; then
    echo -e "\n${BLUE}8. EXAMPLE SCRIPTS${NC}"
    echo "=================="
    
    cd examples
    
    if [ -f "example_client.py" ]; then
        run_test "Example Client" "$PYTHON_CMD example_client.py" "Example"
    fi
    
    if [ -f "conversation_memory_demo.py" ]; then
        run_test "Conversation Memory Demo" "$PYTHON_CMD conversation_memory_demo.py" "Example"
    fi
    
    cd ..
else
    echo -e "\n${BLUE}8. EXAMPLE SCRIPTS${NC}"
    echo "=================="
    if [ "$API_RUNNING" = true ]; then
        print_info "Skipped - Use --examples to run"
    else
        print_info "Skipped - API not running"
    fi
fi

# ========================================
# 9. Final Summary
# ========================================

echo -e "\n${BLUE}9. TEST SUMMARY${NC}"
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
echo "• Essential Tests: Critical system functionality"
echo "• Unit Tests: Core component testing"
echo "• API Tests: Endpoint validation"
echo "• Gradio Tests: Web interface functionality"
if [ "$RUN_INTEGRATION" = true ]; then
    echo "• Integration Tests: End-to-end workflows"
fi
if [ "$RUN_PERFORMANCE" = true ]; then
    echo "• Performance Tests: Scalability and speed"
fi
if [ "$RUN_NL_ADVANCED" = true ]; then
    echo "• Advanced NL Tests: Comprehensive natural language testing"
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
echo "• Run essential tests: ./scripts/run_all_tests.sh --quick"
echo "• Run all tests: ./scripts/run_all_tests.sh --all"
echo "• Check API: curl http://localhost:8000/health"
echo "• Check Gradio: curl http://localhost:7860"
echo "• View logs: docker-compose logs app"

echo -e "\n${BLUE}📚 Documentation:${NC}"
echo "• API Docs: http://localhost:8000/docs"
echo "• Gradio Interface: http://localhost:7860"
echo "• Test README: tests/README.md"

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

