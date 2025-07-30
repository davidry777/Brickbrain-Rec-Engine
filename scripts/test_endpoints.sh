#!/bin/bash

echo "LEGO Recommendation Engine - Endpoint Testing"
echo "============================================="

BASE_URL="http://localhost:8000"
PASSED=0
FAILED=0

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    echo -n "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" -o /tmp/response.json "$BASE_URL$endpoint")
    fi
    
    http_code="${response: -3}"
    
    if [ "$http_code" = "200" ]; then
        echo "✅ PASSED (HTTP $http_code)"
        ((PASSED++))
    else
        echo "❌ FAILED (HTTP $http_code)"
        echo "Response: $(cat /tmp/response.json)"
        ((FAILED++))
    fi
}

# Test endpoints
test_endpoint "Health Check" "GET" "/health" ""
test_endpoint "Themes List" "GET" "/themes" ""
test_endpoint "Set Details" "GET" "/sets/75192-1" ""
test_endpoint "Basic Search" "POST" "/search/sets" '{"query": "space", "limit": 5}'
test_endpoint "Natural Language Search" "POST" "/search/natural" '{"query": "space sets for kids", "top_k": 5}'
test_endpoint "Semantic Similarity" "POST" "/sets/similar/semantic" '{"set_num": "75192-1", "description": "easier to build", "top_k": 5}'
test_endpoint "Recommendations" "POST" "/recommendations" '{"recommendation_type": "content", "set_num": "75192-1", "top_k": 5}'

echo ""
echo "Test Results:"
echo "============"
echo "✅ PASSED: $PASSED"
echo "❌ FAILED: $FAILED"

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! System is production ready."
    exit 0
else
    echo "⚠️  Some tests failed. System needs attention."
    exit 1
fi
