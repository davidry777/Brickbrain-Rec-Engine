#!/bin/bash

# ========================================
# 🧱 LEGO Recommendation Engine 
# Cloud Processing Helper Script
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo -e "${BLUE}🧱 LEGO Recommendation Engine - Cloud Processing Helper${NC}"
echo "======================================================="

# Check if data is ready
if [ ! -f "embeddings/lego_sets_for_cloud.json" ]; then
    print_error "Cloud data not found. Run ./setup_and_start.sh first to prepare data."
    exit 1
fi

DATASET_SIZE=$(wc -l < embeddings/lego_sets_for_cloud.json)
print_info "Dataset ready: $DATASET_SIZE LEGO sets prepared for cloud processing"

echo -e "\n${BLUE}Cloud Platform Options:${NC}"
echo "1. AWS SageMaker (~$0.50, 5-10 minutes)"
echo "2. Azure Machine Learning (~$0.50, 5-10 minutes)" 
echo "3. Google Cloud AI Platform (~$1.00, 5-10 minutes)"
echo "4. Local GPU processing (if available)"
echo ""

read -p "Select platform (1-4): " PLATFORM

case $PLATFORM in
    1)
        echo -e "\n${BLUE}AWS SageMaker Setup:${NC}"
        echo "1. Upload data to S3:"
        echo "   aws s3 cp embeddings/lego_sets_for_cloud.json s3://YOUR-BUCKET/input/"
        echo ""
        echo "2. Update cloud/aws_sagemaker_config.json with your bucket name"
        echo ""
        echo "3. Run SageMaker job:"
        echo "   aws sagemaker create-training-job --cli-input-json file://cloud/aws_sagemaker_config.json"
        echo ""
        echo "4. Download results when complete:"
        echo "   aws s3 sync s3://YOUR-BUCKET/lego-embeddings-output/ embeddings/"
        ;;
    2)
        echo -e "\n${BLUE}Azure Machine Learning Setup:${NC}"
        echo "1. Upload data to Azure ML workspace:"
        echo "   az ml data create --name lego-data --path embeddings/"
        echo ""
        echo "2. Submit job:"
        echo "   az ml job create --file cloud/azure_ml_config.yml"
        echo ""
        echo "3. Download results when complete:"
        echo "   az ml job download --name <job-name> --output-path embeddings/"
        ;;
    3)
        echo -e "\n${BLUE}Google Cloud AI Platform Setup:${NC}"
        echo "1. Upload data to Cloud Storage:"
        echo "   gsutil cp embeddings/lego_sets_for_cloud.json gs://YOUR-BUCKET/input/"
        echo ""
        echo "2. Submit training job:"
        echo "   gcloud ai-platform jobs submit training lego_embeddings \\"
        echo "     --package-path=src/scripts \\"
        echo "     --module-name=cloud_vector_processor \\"
        echo "     --staging-bucket=gs://YOUR-BUCKET"
        echo ""
        echo "3. Download results when complete"
        ;;
    4)
        echo -e "\n${BLUE}Local GPU Processing:${NC}"
        if command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected"
            echo "Running local GPU processing..."
            
            python src/scripts/cloud_vector_processor.py \
                --input embeddings/lego_sets_for_cloud.json \
                --output embeddings/ \
                --batch-size 256
            
            if [ $? -eq 0 ]; then
                print_status "Local GPU processing completed!"
                echo ""
                echo "Loading results into system..."
                docker-compose exec app conda run -n brickbrain-rec python -c "
import sys
sys.path.append('/app/src/scripts')
from lego_nlp_recommeder import NLPRecommender
import psycopg2

conn = psycopg2.connect(
    host='postgres',
    database='brickbrain',
    user='brickbrain',
    password='brickbrain_password',
    port=5432
)

nl = NLPRecommender(conn, use_openai=False)
nl.load_cloud_embeddings('embeddings/embeddings.npy', 'embeddings/metadata.json')
print('✅ Cloud-processed embeddings loaded successfully!')
conn.close()
"
                print_status "System ready with cloud-processed embeddings!"
            fi
        else
            print_warning "No NVIDIA GPU detected. Consider cloud processing for better performance."
        fi
        ;;
    *)
        print_error "Invalid selection"
        exit 1
        ;;
esac

echo -e "\n${GREEN}📚 Additional Resources:${NC}"
echo "• Detailed instructions: cloud/README.md"
echo "• Cost calculator: cloud/cost_calculator.py"
echo "• Performance comparison: cloud/benchmarks.md"

echo -e "\n${GREEN}🔄 After Cloud Processing:${NC}"
echo "1. Download: embeddings.npy, metadata.json, processing_info.json"
echo "2. Load results:"
echo "   python -c \"from src.scripts.lego_nlp_recommeder import NLPRecommender; nl=NLPRecommender(conn); nl.load_cloud_embeddings('embeddings/embeddings.npy', 'embeddings/metadata.json')\""
echo "3. Test with: ./run_all_tests.sh --nl-advanced"

echo -e "\n${BLUE}🎉 Cloud processing setup complete!${NC}"
