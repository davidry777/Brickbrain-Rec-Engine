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
        ORDER BY s.set_num;
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
            # Generate reasons for content-based recommendations
            reasons = self._generate_content_reasons(set_data, set_num)

            recommendations.append(RecommendationResult(
                set_num=set_data['set_num'],
                name=set_data['name'],
                score=float(score),
                reasons=reasons,
                theme_name=set_data['theme_name'],
                year=int(set_data['year']),
                num_parts=int(set_data['num_parts']),
                img_url=set_data['img_url']
            ))

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
    
class CollaborativeFilteringRecommender:
    """
    Collaborative filtering using user ratings and preferences.
    Uses matrix factorization (SVD) for scalable recommendations.
    """
    def __init__(self, dbcon):
        self.dbcon = dbcon
        self.user_item_matrix = None
        self.svd_model = None
        self.user_profiles = {}
        self.item_lookup = {}
        self.reverse_user_profiles = {}
        self.reverse_item_lookup = {}

    def prepare_user_item_matrix(self):
        """
        Prepare user-item matrix for collaborative filtering
        """
        logger.info("########## Preparing user-item matrix ##########")

        # Temporarily creating synthetic user-item data since no real data is provided
        # In a real scenario, this would be loaded from the database
        self._create_synthetic_user_data()

        # Load user ratings
        query = """
        SELECT user_id, set_num, rating, interaction_type, created_at
        FROM user_iteractions
        WHERE rating IS NOT NULL
        ORDER BY user_id, set;
        """

        try:
            ratings_df = pd.read_sql(query, self.dbcon)
        except Exception as e:
            ratings_df = self.synthetic_ratings

        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='set_num', 
            values='rating', 
            fill_value=0
        )

        # Create lookup dictionaries
        self.user_lookup = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.item_lookup = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        self.reverse_user_lookup = {idx: user for user, idx in self.user_lookup.items()}
        self.reverse_item_lookup = {idx: item for item, idx in self.item_lookup.items()}
        
        logger.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
    def _create_synthetic_user_data(self):
        """Create synthetic user data for testing (remove in production)"""
        np.random.seed(42)
        
        # Get some sets from database
        sets_query = "SELECT set_num FROM sets LIMIT 100"
        sets_df = pd.read_sql(sets_query, self.db_connection)
        set_nums = sets_df['set_num'].tolist()
        
        # Create synthetic ratings
        synthetic_data = []
        for user_id in range(1, 51):  # 50 synthetic users
            # Each user rates 10-30 sets randomly
            num_ratings = np.random.randint(10, 31)
            rated_sets = np.random.choice(set_nums, num_ratings, replace=False)
            
            for set_num in rated_sets:
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])  # Biased toward positive
                synthetic_data.append({
                    'user_id': user_id,
                    'set_num': set_num,
                    'rating': rating,
                    'interaction_type': 'rating',
                    'created_at': datetime.now()
                })
        
        self.synthetic_ratings = pd.DataFrame(synthetic_data)

    def train_svd_model(self, n_components: int = 50):
        """
        Train SVD model for collaborative filtering

        :param n_components: Number of latent factors
        """
        if self.user_item_matrix is None:
            self.prepare_user_item_matrix()

        logger.info("########## Training SVD model ##########")
        
        # Create SVD model
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        item_factors = self.svd_model.components_.T
        
        # Store factors for later use
        self.user_factors = user_factors
        self.item_factors = item_factors

        logger.info(f"Trained SVD model with {n_components} components")

    def get_recommendations(self, user_id: str, top_k: int = 10) -> List[RecommendationResult]:
        """
        Get recommendations for a user based on collaborative filtering

        :param user_id: The user ID to get recommendations for
        :param top_k: Number of recommendations to return
        :return: List of RecommendationResult objects
        """
        if self.svd_model is None:
            self.train_svd_model()

        if user_id not in self.user_lookup:
            logger.warning(f"User {user_id} not found in user lookup")
            return self._cold_start_recommendations(top_k)

        user_idx = self.user_lookup[user_id]
        
        # Get user's latent factors
        user_factors = self.user_factors[user_idx].reshape(1, -1)
        
        # Calculate scores for all items
        scores = np.dot(user_factors, self.item_factors).flatten()
        
        # Get top K item indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            set_num = self.reverse_item_lookup[idx]
            score = scores[idx]
            reasons = [f"Recommended based on collaborative filtering with score {score:.2f}"]
            
            # Fetch set details from database
            set_query = f"SELECT * FROM sets WHERE set_num = '{set_num}'"
            set_data = pd.read_sql(set_query, self.dbcon).iloc[0]
            
            recommendations.append(RecommendationResult(
                set_num=set_data['set_num'],
                name=set_data['name'],
                score=float(score),
                reasons=reasons,
                theme_name=set_data['theme_name'],
                year=int(set_data['year']),
                num_parts=int(set_data['num_parts']),
                img_url=set_data['img_url']
            ))

        return recommendations
    
    def _cold_start_recommendations(self, top_k: int) -> List[RecommendationResult]:
        """Handle cold start problem for new users"""
        # Recommend popular items (highest average ratings)
        popular_query = """
        SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url, t.name as theme_name,
               AVG(ui.rating) as avg_rating, COUNT(ui.rating) as rating_count
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN user_interactions ui ON s.set_num = ui.set_num
        WHERE ui.rating IS NOT NULL
        GROUP BY s.set_num, s.name, s.year, s.num_parts, s.img_url, t.name
        HAVING COUNT(ui.rating) >= 5
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT %s
        """
        
        try:
            popular_sets = pd.read_sql(popular_query, self.db_connection, params=[top_k])
            recommendations = []
            
            for _, row in popular_sets.iterrows():
                recommendations.append(RecommendationResult(
                    set_num=row['set_num'],
                    name=row['name'],
                    score=float(row['avg_rating']),
                    reasons=[f"Popular choice (avg rating: {row['avg_rating']:.1f})"],
                    theme_name=row['theme_name'],
                    year=int(row['year']),
                    num_parts=int(row['num_parts']),
                    img_url=row['img_url']
                ))
                
            return recommendations
        except:
            # Fallback to recent popular sets
            return self._get_recent_popular_sets(top_k)
    
    def _get_set_details(self, set_num: str) -> Optional[Dict]:
        """Get set details from database"""
        query = """
        SELECT s.name, s.year, s.num_parts, s.img_url, t.name as theme_name
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        WHERE s.set_num = %s
        """
        
        try:
            result = pd.read_sql(query, self.db_connection, params=[set_num])
            if not result.empty:
                return result.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error getting set details for {set_num}: {e}")
        
        return None
    
    def _get_recent_popular_sets(self, top_k: int) -> List[RecommendationResult]:
        """Fallback method to get recent popular sets"""
        query = """
        SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url, t.name as theme_name
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        WHERE s.year >= 2020 AND s.num_parts BETWEEN 100 AND 1000
        ORDER BY s.year DESC, s.num_parts DESC
        LIMIT %s
        """
        
        try:
            popular_sets = pd.read_sql(query, self.db_connection, params=[top_k])
            recommendations = []
            
            for _, row in popular_sets.iterrows():
                recommendations.append(RecommendationResult(
                    set_num=row['set_num'],
                    name=row['name'],
                    score=0.8,  # Default score
                    reasons=["Popular recent set"],
                    theme_name=row['theme_name'] or 'Unknown',
                    year=int(row['year']),
                    num_parts=int(row['num_parts']),
                    img_url=row['img_url']
                ))
                
            return recommendations
        except Exception as e:
            logger.error(f"Error getting popular sets: {e}")
            return []
        
class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    """
    
    def __init__(self, dbcon):
        self.dbcon = dbcon
        self.content_recommender = ContentBasedRecommender(dbcon)
        self.collaborative_recommender = CollaborativeFilteringRecommender(dbcon)

        # Weights for combining recommendations
        self.content_weight = 0.4
        self.collaborative_weight = 0.6

    def get_recommendations(self, user_id: Optional[int] = None, liked_set: Optional[str] = None, top_k: int = 10) -> List[RecommendationResult]:
        """
        Get hybrid recommendations for a user

        :param user_id: The user ID to get recommendations for
        :param liked_set: A set number that the user likes (for content-based filtering)
        :param top_k: Number of recommendations to return
        :return: List of RecommendationResult objects
        """
        content_recs = []
        collaborative_recs = []
        
        # Get content-based recommendations
        if liked_set:
            content_recs = self.content_recommender.get_similar_sets(liked_set, top_k * 2)
        
        # Get collaborative filtering recommendations
        if user_id:
            collaborative_recs = self.collaborative_recommender.get_recommendations(user_id, top_k * 2)
        
        # If only one type is available, return that
        if not content_recs and collaborative_recs:
            return collaborative_recs[:top_k]
        elif content_recs and not collaborative_recs:
            return content_recs[:top_k]
        elif not content_recs and not collaborative_recs:
            # Cold start - return popular sets
            return self.collaborative_recommender._get_recent_popular_sets(top_k)
        
        # Combine recommendations
        return self._combine_recommendations(content_recs, collaborative_recs, top_k)
    
    def _combine_recommendations(self, content_recs: List[RecommendationResult], collaborative_recs: List[RecommendationResult], top_k: int) -> List[RecommendationResult]:
        """
        Combine and rank recommendations from both approaches
        :param content_recs: Content-based recommendations
        :param collaborative_recs: Collaborative filtering recommendations
        :param top_k: Number of recommendations to return
        :return: Combined and ranked list of RecommendationResult objects
        """
        
        # Create dictionaries for quick lookup
        content_dict = {rec.set_num: rec for rec in content_recs}
        collaborative_dict = {rec.set_num: rec for rec in collaborative_recs}
        
        # Get all unique set numbers
        all_sets = set(content_dict.keys()) | set(collaborative_dict.keys())
        
        combined_recs = []
        
        for set_num in all_sets:
            content_rec = content_dict.get(set_num)
            collab_rec = collaborative_dict.get(set_num)
            
            # Calculate hybrid score
            content_score = content_rec.score if content_rec else 0
            collab_score = collab_rec.score if collab_rec else 0
            
            hybrid_score = (
                self.content_weight * content_score + 
                self.collaborative_weight * collab_score
            )
            
            # Use the recommendation with more complete information
            base_rec = content_rec if content_rec else collab_rec
            
            # Combine reasons from both approaches
            reasons = []
            if content_rec:
                reasons.extend([f"Content: {reason}" for reason in content_rec.reasons])
            if collab_rec:
                reasons.extend([f"Community: {reason}" for reason in collab_rec.reasons])
            
            combined_recs.append(RecommendationResult(
                set_num=set_num,
                name=base_rec.name,
                score=hybrid_score,
                reasons=reasons,
                theme_name=base_rec.theme_name,
                year=base_rec.year,
                num_parts=base_rec.num_parts,
                img_url=base_rec.img_url
            ))
        
        # Sort by hybrid score and return top k
        combined_recs.sort(key=lambda x: x.score, reverse=True)
        return combined_recs[:top_k]
    
    def set_weights(self, content_weight: float, collaborative_weight: float):
        """Adjust weights for combining recommendations"""
        total_weight = content_weight + collaborative_weight
        self.content_weight = content_weight / total_weight
        self.collaborative_weight = collaborative_weight / total_weight
        
        logger.info(f"Updated weights - Content: {self.content_weight:.2f}, Collaborative: {self.collaborative_weight:.2f}")

    # Example usage and testing functions
def test_recommendation_engine(db_connection):
    """Test the recommendation engine with sample data"""
    
    # Initialize hybrid recommender
    recommender = HybridRecommender(db_connection)
    
    # Test content-based recommendations
    print("Testing Content-Based Recommendations:")
    print("=" * 50)
    
    # Get a sample set to test with
    sample_query = "SELECT set_num, name FROM sets WHERE num_parts BETWEEN 200 AND 500 LIMIT 1"
    sample_set = pd.read_sql(sample_query, db_connection)
    
    if not sample_set.empty:
        test_set = sample_set.iloc[0]['set_num']
        print(f"Finding sets similar to: {sample_set.iloc[0]['name']} ({test_set})")
        
        content_recs = recommender.content_recommender.get_similar_sets(test_set, 5)
        
        for i, rec in enumerate(content_recs, 1):
            print(f"{i}. {rec.name} ({rec.set_num})")
            print(f"   Score: {rec.score:.3f}")
            print(f"   Theme: {rec.theme_name}")
            print(f"   Reasons: {', '.join(rec.reasons)}")
            print()
    
    # Test collaborative filtering
    print("Testing Collaborative Filtering:")
    print("=" * 50)
    
    collab_recs = recommender.collaborative_recommender.get_user_recommendations(1, 5)
    
    for i, rec in enumerate(collab_recs, 1):
        print(f"{i}. {rec.name} ({rec.set_num})")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Theme: {rec.theme_name}")
        print(f"   Reasons: {', '.join(rec.reasons)}")
        print()
    
    # Test hybrid recommendations
    print("Testing Hybrid Recommendations:")
    print("=" * 50)
    
    if not sample_set.empty:
        hybrid_recs = recommender.get_hybrid_recommendations(
            user_id=1, 
            liked_set=test_set, 
            top_k=5
        )
        
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"{i}. {rec.name} ({rec.set_num})")
            print(f"   Hybrid Score: {rec.score:.3f}")
            print(f"   Theme: {rec.theme_name}")
            print(f"   Reasons: {', '.join(rec.reasons)}")
            print()

if __name__ == "__main__":
    import psycopg2

    # Database connection parameters
    db_params = {
        'dbname': 'brickbrain',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }

    try:
        connection = psycopg2.connect(**db_params)
        test_recommendation_engine(connection)
        connection.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure your PostgreSQL container is running and data is loaded.")