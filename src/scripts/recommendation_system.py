import pandas as pd
import numpy as np
import logging
import pickle
import os

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecommendationResult:
    """Recommendation result data class"""
    set_num: str
    name: str
    score: float
    reasons: List[str]
    theme_name: str
    year: int
    num_parts: int
    img_url: Optional[str] = None

@dataclass
class UserProfile:
    """User profile data class"""
    user_id: str
    preferred_themes: List[int]
    preferred_piece_ranges: List[Tuple[int, int]]
    preferred_years: List[int]
    complexity_preference: str  # "simple", "moderate", "complex"
    avg_rating: float
    total_ratings: int

class ContentBasedRecommender:
    """
    Content-based filtering using set attributes like theme, piece count, year, and complexity.
    """

    def __init__(self, dbcon):
        self.dbcon = dbcon
        self.set_feat = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer()
        self.feat_matrix = None
        self.set_lookup = {}

    def prepare_features(self):
        """
        Prepare and engineer features for content-based filtering
        """
        logger.info("########## Preparing content-based features ##########")

        # Load set data with theme info
        query = """
        SELECT 
            s.set_num,
            s.name,
            s.year,
            s.theme_id,
            s.num_parts,
            s.img_url,
            t.name as theme_name,
            t.parent_id as parent_theme_id,
            COUNT(DISTINCT ip.part_num) as unique_parts,
            COUNT(DISTINCT ip.color_id) as unique_colors,
            AVG(CASE WHEN ip.is_spare THEN 0 ELSE ip.quantity END) as avg_part_quantity
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN inventories i ON s.set_num = i.set_num
        LEFT JOIN inventory_parts ip ON i.id = ip.inventory_id
        WHERE s.num_parts > 0
        GROUP BY s.set_num, s.name, s.year, s.theme_id, s.num_parts, s.img_url, t.name, t.parent_id
        ORDER BY s.set_num
        """

        self.set_feat = pd.read_sql(query, self.dbcon)

        # Handling missing features
        self.set_feat['num_parts'] = self.set_feat['num_parts'].fillna(0)
        self.set_feat['unique_parts'] = self.set_feat['unique_parts'].fillna(0)
        self.set_feat['unique_colors'] = self.set_feat['unique_colors'].fillna(0)
        self.set_feat['avg_part_quantity'] = self.set_feat['avg_part_quantity'].fillna(0)

        # TODO: Create feature matrix for similarity calculations

        # Creating lookup dictionary
        self.set_lookup = dict(zip(
            range(len(self.set_feat)),
            self.set_feat['set_num']
        ))

        logger.info(f"--------- Prepared features for {len(self.set_features)} sets ---------")

    def get_similar_sets(self, set_num: str, top_k: int = 10) -> List[RecommendationResult]:
        """
        Get sets similar to the given set

        :param set_num: The set number to find similar sets for
        :param top_k: Number of similar sets to return
        :return: List of RecommendationResult objects
        """
        if self.feat_matrix is None:
            self.prepare_features()

        try:
            target_idx = self.set_feat[self.set_feat['set_num'] == set_num]
        except IndexError:
            logger.error(f"Set {set_num} not found")
            return []
        
        # Calculate similarities
        target_feats = self.feat_matrix[target_idx].reshape(1, -1)
        similarities = cosine_similarity(target_feats, self.feat_matrix)[0]

        # Extract top similar sets (excluding target set)
        sim_idxs = np.argsort(similarities)[::-1][1:top_k+1]

        recommendations = []
        for i in sim_idxs:
            set_data = self.set_feat.iloc[i]
            score = similarities[i]

            # TODO: Generate reasons for recommendation

        return recommendations

