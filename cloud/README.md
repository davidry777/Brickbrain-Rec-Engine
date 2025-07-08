# Cloud Vector Processing for LEGO Recommendation Engine

This directory contains scripts and configurations for processing LEGO embeddings on cloud platforms.

## Why Use Cloud Processing?

### Performance Benefits
- **10-100x faster**: GPU acceleration vs local CPU
- **Scalable**: Process any dataset size
- **Parallel**: Multiple workers for large datasets
- **Cost-effective**: Pay only for compute time used

### Example Processing Times
| Dataset Size | Local CPU | Cloud GPU (V100) | Speedup |
|-------------|-----------|------------------|---------|
| 1,000 sets  | 5 minutes | 30 seconds      | 10x     |
| 10,000 sets | 50 minutes| 3 minutes       | 17x     |
| 25,000 sets | 2+ hours  | 7 minutes       | 18x     |

## Quick Start

### 1. Prepare Data Locally
```bash
# Generate cloud-ready data file
python src/scripts/lego_nlp_recommeder.py --prepare-cloud-data

# This creates: embeddings/lego_sets_for_cloud.json
```

### 2. Choose Your Cloud Platform

#### AWS SageMaker
```bash
# Upload to S3 and run SageMaker job
aws s3 cp embeddings/lego_sets_for_cloud.json s3://your-bucket/input/
aws sagemaker create-training-job --cli-input-json file://cloud/aws_sagemaker_config.json
```

#### Azure Machine Learning
```bash
# Upload and run Azure ML job
az ml job create --file cloud/azure_ml_config.yml
```

#### Google Cloud AI Platform
```bash
# Run on Google Cloud
gcloud ai-platform jobs submit training lego_embeddings \
    --package-path=src/scripts \
    --module-name=cloud_vector_processor \
    --config=cloud/gcp_config.yaml
```

### 3. Download Results
After processing, download the generated files:
- `embeddings.npy`: Pre-computed embeddings
- `metadata.json`: Set metadata
- `processing_info.json`: Processing statistics

### 4. Load in Local System
```python
from src.scripts.lego_nlp_recommeder import NLPRecommender

# Initialize with cloud-processed embeddings
nl_recommender = NLPRecommender(conn, use_openai=False)
nl_recommender.load_cloud_embeddings(
    "embeddings/embeddings.npy",
    "embeddings/metadata.json"
)
```

## Cloud Platform Configurations

### AWS SageMaker
- **Instance**: `ml.p3.2xlarge` (V100 GPU)
- **Cost**: ~$3.06/hour (only during processing)
- **Estimated time**: 5-10 minutes for full dataset
- **Total cost**: ~$0.50-$1.00 for complete processing

### Azure Machine Learning
- **Instance**: `Standard_NC6s_v3` (V100 GPU)
- **Cost**: ~$3.06/hour
- **Estimated time**: 5-10 minutes for full dataset

### Google Cloud AI Platform
- **Instance**: `n1-standard-4` + `nvidia-tesla-v100`
- **Cost**: ~$2.48/hour + $2.48/hour (GPU)
- **Total**: ~$4.96/hour during processing

## Development Workflow

### Local Development
```bash
# Quick setup with limited data (for development)
./setup_and_start.sh
python -c "
from src.scripts.lego_nlp_recommeder import NLPRecommender
nl = NLPRecommender(conn, use_openai=False)
nl.prep_vectorDB(limit_sets=100)  # Fast local testing
"
```

### Production Deployment
```bash
# Prepare for cloud processing
python -c "
from src.scripts.lego_nlp_recommeder import NLPRecommender
nl = NLPRecommender(conn, use_openai=False)
nl.prep_vectorDB(use_cloud=True)  # Prepares cloud data
"

# Process on cloud (see platform-specific instructions above)

# Deploy with cloud-processed embeddings
python -c "
nl.load_cloud_embeddings('embeddings/embeddings.npy', 'embeddings/metadata.json')
"
```

## Cost Analysis

### One-time Processing Costs
- **AWS SageMaker**: $0.50-$1.00 (5-10 minutes on ml.p3.2xlarge)
- **Azure ML**: $0.50-$1.00 (5-10 minutes on Standard_NC6s_v3)
- **Google Cloud**: $0.80-$1.60 (5-10 minutes with GPU)

### Ongoing Costs
- **Storage**: <$0.01/month for embeddings files
- **Compute**: Only API serving costs (unchanged)

### ROI Calculation
- **Local processing time saved**: 1-2 hours → 5-10 minutes
- **Development productivity**: Faster iteration cycles
- **Production reliability**: Professional ML infrastructure
- **Scalability**: Handle any dataset size

## Advanced Options

### Custom Models
```python
# Use different embedding models
processor = CloudVectorProcessor(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Higher quality
)
```

### Batch Processing
```python
# Optimize for large datasets
processor.process_lego_data(
    input_file="lego_sets_for_cloud.json",
    output_dir="results/",
    batch_size=1024  # Larger batches on powerful GPUs
)
```

### Multi-language Support
```python
# Process in multiple languages
processor = CloudVectorProcessor(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## Monitoring and Debugging

### Processing Logs
Cloud platforms provide detailed logs:
- Processing rate (items/second)
- Memory usage
- GPU utilization
- Error detection

### Quality Validation
```python
# Validate embeddings quality
from src.scripts.validate_embeddings import EmbeddingValidator

validator = EmbeddingValidator()
validator.check_embedding_quality("embeddings/embeddings.npy")
validator.test_search_relevance(nl_recommender)
```

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch_size
2. **Slow Processing**: Ensure GPU instance type
3. **Upload Failures**: Check file permissions and network
4. **Model Loading**: Verify model name and availability

### Support
- Check cloud platform documentation
- Monitor processing logs
- Validate input data format
- Test with smaller datasets first
