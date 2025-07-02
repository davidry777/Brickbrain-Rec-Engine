#!/bin/bash
# Test Runner Script for LEGO Recommendation System
# This script runs all tests in the correct order and provides a summary

echo "🧱 LEGO Recommendation System - Test Suite Runner"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
        if [[ "$test_name" == *"Simple"* ]]; then
            grep -E "✅|🎯|🎉" /tmp/test_output | head -3
        fi
    else
        echo -e "   ${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "   Error details:"
        tail -n 5 /tmp/test_output | sed 's/^/     /'
    fi
}

# Check prerequisites
echo -e "\n🔍 Checking Prerequisites..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Warning: Docker not accessible. Database tests may fail.${NC}"
fi

# Check if API is running
# Try to check API with different methods depending on environment
if [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
    # We're in the container, use python to check
    if conda run -n brickbrain-rec python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is running${NC}"
        API_RUNNING=true
    else
        echo -e "${YELLOW}⚠️  API not running. Integration tests will be skipped.${NC}"
        API_RUNNING=false
    fi
elif curl -s http://localhost:8000/health > /dev/null 2>&1; then
    # We're on the host, use curl
    echo -e "${GREEN}✅ API is running${NC}"
    API_RUNNING=true
else
    echo -e "${YELLOW}⚠️  API not running. Integration tests will be skipped.${NC}"
    API_RUNNING=false
fi

# Unit Tests
echo -e "\n🔧 UNIT TESTS"
echo "=============="
cd tests/unit

# Check if we're in a conda environment, if so use conda run
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
    run_test "Database Tests" "conda run -n brickbrain-rec python test_database.py" "Unit Test"
    run_test "Recommendation Tests" "conda run -n brickbrain-rec python test_recommendations.py" "Unit Test"
else
    run_test "Database Tests" "python test_database.py" "Unit Test"
    run_test "Recommendation Tests" "python test_recommendations.py" "Unit Test"
fi
cd ../..

# Integration Tests (only if API is running)
if [ "$API_RUNNING" = true ]; then
    echo -e "\n🔗 INTEGRATION TESTS"
    echo "===================="
    cd tests/integration
    
    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
        run_test "Simple Production Test" "conda run -n brickbrain-rec python production_test_simple.py" "Integration Test"
        run_test "Final Validation" "conda run -n brickbrain-rec python final_validation.py" "Integration Test"
        run_test "Production Readiness" "conda run -n brickbrain-rec python validate_production_readiness.py" "Integration Test"
    else
        run_test "Simple Production Test" "python production_test_simple.py" "Integration Test"
        run_test "Final Validation" "python final_validation.py" "Integration Test"
        run_test "Production Readiness" "python validate_production_readiness.py" "Integration Test"
    fi
    cd ../..
else
    echo -e "\n🔗 INTEGRATION TESTS"
    echo "===================="
    echo -e "${YELLOW}⚠️  Skipped - API not running${NC}"
fi

# Performance Tests (optional - takes longer)
if [ "$1" = "--performance" ] || [ "$1" = "--all" ]; then
    echo -e "\n⚡ PERFORMANCE TESTS"
    echo "===================="
    cd tests/performance
    
    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
        run_test "Scalability Test" "conda run -n brickbrain-rec python production_scalability_test.py" "Performance Test"
    else
        run_test "Scalability Test" "python production_scalability_test.py" "Performance Test"
    fi
    cd ../..
else
    echo -e "\n⚡ PERFORMANCE TESTS"
    echo "===================="
    echo -e "${YELLOW}⚠️  Skipped - Use --performance or --all to run${NC}"
fi

# Examples (if API is running)
if [ "$API_RUNNING" = true ] && ([ "$1" = "--examples" ] || [ "$1" = "--all" ]); then
    echo -e "\n📱 EXAMPLES"
    echo "==========="
    cd examples
    
    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ] || [ -f "/opt/conda/envs/brickbrain-rec/bin/python" ]; then
        run_test "Example Client" "conda run -n brickbrain-rec python example_client.py" "Example"
    else
        run_test "Example Client" "python example_client.py" "Example"
    fi
    cd ..
else
    echo -e "\n📱 EXAMPLES"
    echo "==========="
    echo -e "${YELLOW}⚠️  Skipped - Use --examples or --all to run${NC}"
fi

# Final Summary
echo -e "\n" 
echo "================================================="
echo "🎯 TEST SUMMARY"
echo "================================================="

if [ $TOTAL_TESTS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No tests were run${NC}"
    exit 1
fi

echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "Success Rate: $SUCCESS_RATE%"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! SYSTEM IS READY!${NC}"
    exit 0
elif [ $SUCCESS_RATE -ge 75 ]; then
    echo -e "\n${YELLOW}⚠️  MOSTLY WORKING - Minor issues detected${NC}"
    exit 0
else
    echo -e "\n${RED}❌ CRITICAL ISSUES DETECTED - Fix before deployment${NC}"
    exit 1
fi
