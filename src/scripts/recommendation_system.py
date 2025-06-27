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

        # Feature engineering
        self.set_feat['complexity_score'] = self._calculate_complexity_score()
        self.set_feat['age_score'] = self._calculate_age_score()
        self.set_feat['size_category'] = self._categorize_by_size()

        # Create feature matrix for similarity calculations
        self._create_feature_matrix()

        # Creating lookup dictionary
        self.set_lookup = dict(zip(
            range(len(self.set_feat)),
            self.set_feat['set_num']
        ))

        logger.info(f"--------- Prepared features for {len(self.set_feat)} sets ---------")

    def _calculate_complexity_score(self) -> pd.Series:
        """Calculate complexity score based on parts, unique parts, and colors"""
        complexity = (
            0.4 * self.set_feat['num_parts'] / self.set_feat['num_parts'].max() +
            0.3 * self.set_feat['unique_parts'] / self.set_feat['unique_parts'].max() +
            0.2 * self.set_feat['unique_colors'] / self.set_feat['unique_colors'].max() +
            0.1 * self.set_feat['avg_part_quantity'] / self.set_feat['avg_part_quantity'].max()
        )
        return complexity.fillna(0)
    
    def _calculate_age_score(self) -> pd.Series:
        """Calculate age/vintage score"""
        current_year = datetime.now().year
        age = current_year - self.set_feat['year']
        # Normalize age score (newer sets get higher scores)
        return 1 - (age / age.max())
    
    def _categorize_by_size(self) -> pd.Series:
        """Categorize sets by size"""
        def size_category(num_parts):
            if num_parts < 100:
                return 'small'
            elif num_parts < 500:
                return 'medium'
            elif num_parts < 1000:
                return 'large'
            else:
                return 'xl'
        
        return self.set_feat['num_parts'].apply(size_category)

    def _create_feature_matrix(self):
        """
        Create normalized feature matrix for similarity calculations
        """
        numeric_feats = [
            'num_parts', 'unique_parts', 'unique_colors', 
            'complexity_score', 'age_score', 'theme_id'
        ]
        feat_mat = self.set_feat[numeric_feats].copy()

        # Scale numerical features
        scaled_feats = self.scaler.fit_transform(feat_mat)
        # Add categorical features (one-hot encoder)
        size_dummies = pd.get_dummies(self.set_feat['size_category'], prefix='size')

        # Combine all features
        self.feat_matrix = np.hstack([scaled_feats, size_dummies.values])
        logger.info(f"Feature matrix shape: {self.feat_matrix.shape}")

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
    
    def _generate_content_reasons(self, similar_set, target_set_num) -> List[str]:
        """
        Generate reasons for content-based recommendations
        """
        target_set = self.set_feat[self.set_feat['set_num'] == target_set_num].iloc[0]
        reasons = []

        # Theme similarity
        if similar_set['theme_id'] == target_set['theme_id']:
            reasons.append(f"Same theme: {similar_set['theme_name']}")
        
        # Size similarity
        if similar_set['size_category'] == target_set['size_category']:
            reasons.append(f"Same size category: {similar_set['size_category']}")

        # Complexity similarity
        complexity_score = abs(similar_set['complexity_score'] - target_set['complexity_score'])
        if complexity_score < 0.2:  # Threshold for similarity
            reasons.append("Similar complexity level")

        # Year similarity
        year_diff = abs(similar_set['year'] - target_set['year'])
        if year_diff <= 3:  # Threshold for year similarity
            reasons.append(f"From similar era")

        return reasons
    
