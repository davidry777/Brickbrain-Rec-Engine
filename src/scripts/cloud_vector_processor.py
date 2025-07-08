#!/usr/bin/env python3
"""
Cloud Vector Processing Script for LEGO Recommendation Engine
This script is designed to run on cloud ML platforms like AWS SageMaker or Azure ML
to efficiently process large datasets and generate embeddings.
"""

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
import logging
from pathlib import Path
import torch
from typing import List, Dict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudVectorProcessor:
    """Cloud-based vector processing for LEGO sets"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the cloud processor
        
        :param model_name: Hugging Face model name for embeddings
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing model {model_name} on {self.device}")
        
        # Load model - on cloud this will be much faster with GPU
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Model loaded successfully")
    
    def process_lego_data(self, input_file: str, output_dir: str, batch_size: int = 512):
        """
        Process LEGO data and generate embeddings
        
        :param input_file: Path to JSON file with LEGO set data
        :param output_dir: Directory to save outputs
        :param batch_size: Batch size for processing (larger on cloud with GPU)
        """
        logger.info(f"Processing LEGO data from {input_file}")
        
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} LEGO sets")
        
        # Extract texts for embedding
        texts = [item['content'] for item in data]
        metadata = [item['metadata'] for item in data]
        
        # Process in batches for memory efficiency
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} items)")
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            # Log progress
            elapsed = time.time() - start_time
            items_processed = min(i + batch_size, len(texts))
            rate = items_processed / elapsed
            eta = (len(texts) - items_processed) / rate if rate > 0 else 0
            
            logger.info(f"Processed {items_processed}/{len(texts)} items "
                       f"(Rate: {rate:.1f} items/sec, ETA: {eta:.1f}s)")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Save processing info
        info = {
            "model_name": self.model_name,
            "num_documents": len(texts),
            "embedding_dimension": embeddings.shape[1],
            "processing_time": time.time() - start_time,
            "device_used": self.device,
            "batch_size": batch_size
        }
        
        info_file = output_path / "processing_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved processing info to {info_file}")
        
        total_time = time.time() - start_time
        logger.info(f"Processing complete! Total time: {total_time:.2f}s")
        logger.info(f"Average rate: {len(texts)/total_time:.1f} items/sec")
        
        return embeddings, metadata, info

def main():
    parser = argparse.ArgumentParser(description="Cloud Vector Processing for LEGO Recommendation Engine")
    parser.add_argument("--input", required=True, help="Input JSON file with LEGO data")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    parser.add_argument("--batch-size", type=int, default=512, 
                       help="Batch size for processing (use larger values on GPU)")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = CloudVectorProcessor(model_name=args.model)
    processor.process_lego_data(
        input_file=args.input,
        output_dir=args.output,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
