import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import json
import re
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# HuggingFace imports
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, pipeline,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer

# LangChain imports (for compatibility with existing system)
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NLQueryResult:
    """Result from natural language query processing"""
    intent: str
    filters: Dict[str, Any]
    confidence: float
    extracted_entities: Dict[str, Any]
    semantic_query: str

@dataclass
class ConversationContext:
    """Context for maintaining conversation state"""
    user_preferences: Dict[str, Any]
    previous_recommendations: List[Dict[str, Any]]
    current_session_queries: List[str]
    follow_up_context: Dict[str, Any]
    conversation_summary: str

class SearchFilters(BaseModel):
    """Pydantic model for parsed search filters"""
    themes: Optional[List[str]] = Field(None, description="LEGO themes")
    min_pieces: Optional[int] = Field(None, description="Minimum number of pieces")
    max_pieces: Optional[int] = Field(None, description="Maximum number of pieces")
    min_age: Optional[int] = Field(None, description="Minimum age recommendation")
    max_age: Optional[int] = Field(None, description="Maximum age recommendation")
    year_range: Optional[List[int]] = Field(None, description="Year range [start, end]")
    price_range: Optional[List[float]] = Field(None, description="Price range [min, max]")

class HuggingFaceConversationMemory:
    """Conversation memory optimized for HuggingFace models"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations = []
        self.context_summaries = []
    
    def add_interaction(self, user_message: str, assistant_response: str, context: Dict = None):
        """Add a conversation interaction"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "context": context or {}
        }
        self.conversations.append(interaction)
        
        # Keep only recent conversations
        if len(self.conversations) > self.max_history:
            self.conversations = self.conversations[-self.max_history:]
    
    def get_recent_context(self, n_interactions: int = 3) -> str:
        """Get recent conversation context as formatted string"""
        if not self.conversations:
            return ""
        
        recent = self.conversations[-n_interactions:]
        context_parts = []
        
        for interaction in recent:
            context_parts.append(f"User: {interaction['user_message']}")
            context_parts.append(f"Assistant: {interaction['assistant_response'][:100]}...")
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation"""
        if not self.conversations:
            return "No previous conversation."
        
        # Simple rule-based summary
        themes_mentioned = set()
        user_preferences = []
        
        for conv in self.conversations[-5:]:  # Last 5 interactions
            user_msg = conv['user_message'].lower()
            
            # Extract themes mentioned
            lego_themes = ['star wars', 'harry potter', 'technic', 'city', 'creator', 
                          'friends', 'ninjago', 'architecture', 'ideas', 'castle']
            for theme in lego_themes:
                if theme in user_msg:
                    themes_mentioned.add(theme.title())
            
            # Extract preferences
            if any(word in user_msg for word in ['like', 'love', 'prefer', 'favorite']):
                user_preferences.append(user_msg)
        
        summary_parts = []
        if themes_mentioned:
            summary_parts.append(f"Interested in: {', '.join(themes_mentioned)}")
        if user_preferences:
            summary_parts.append(f"Recent preference: {user_preferences[-1][:50]}...")
        
        return "; ".join(summary_parts) if summary_parts else "General LEGO interest"
    
    def clear(self):
        """Clear conversation history"""
        self.conversations = []
        self.context_summaries = []

class HuggingFaceNLPRecommender:
    """
    HuggingFace-based Natural Language Processing for LEGO recommendations
    
    This class replaces the Ollama-based implementation with optimized HuggingFace models
    for better performance and local deployment capabilities.
    """
    
    def __init__(self, dbcon, use_quantization: bool = True, device: str = None):
        """
        Initialize the HuggingFace NLP Recommender
        
        Args:
            dbcon: Database connection
            use_quantization: Whether to use 4-bit quantization for memory efficiency
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.dbconn = dbcon
        self.device = device or self._detect_device()
        self.use_quantization = use_quantization and self.device != 'cpu'
        
        logger.info(f"Initializing HuggingFace NLP Recommender on device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Initialize conversation memory
        self.conversation_memory = HuggingFaceConversationMemory()
        
        # User context storage
        self.user_context = {
            'preferences': {},
            'previous_searches': [],
            'liked_sets': [],
            'disliked_sets': [],
            'conversation_session': datetime.now().isoformat()
        }
        
        # Vector store for semantic search
        self.vectorstore = None
        self.embedding_model = None
        
        # Initialize vector database
        self._init_vector_store()
        
        # Load themes and categories from database
        self.lego_themes = {}
        self.theme_hierarchy = {}
        self.interest_categories = {}
        self.theme_cache_timestamp = None
        
        # Try to load themes from database, fall back to hardcoded if it fails
        try:
            self._load_themes_from_database()
        except Exception as e:
            logger.warning(f"Failed to load themes from database during initialization: {e}")
            self._load_fallback_themes()
        
        # LEGO-specific intents and patterns
        self.intents = {
            'search': ['find', 'search', 'looking for', 'show me', 'want', 'need', 'browse'],
            'recommend_similar': ['similar to', 'like', 'comparable', 'alternative to', 'reminds me of'],
            'gift_recommendation': ['gift', 'present', 'birthday', 'christmas', 'holiday', 
                                  'for my nephew', 'for my son', 'for my daughter', 'for a child'],
            'collection_advice': ['should i buy', 'worth it', 'good investment', 'collection', 
                                'for my collection', 'complete my collection'],
            'budget_conscious': ['cheap', 'affordable', 'budget', 'under', 'less than', 'save money'],
            'advanced_builder': ['complex', 'challenging', 'detailed', 'advanced', 'expert level'],
            'beginner_friendly': ['easy', 'simple', 'beginner', 'starter', 'first time']
        }
        
        self.is_initialized = True
        logger.info("HuggingFace NLP Recommender initialized successfully")
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _init_models(self):
        """Initialize HuggingFace models for different NLP tasks"""
        
        # Quantization config for memory efficiency
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # 1. Intent Classification Model
        # Using a lightweight BERT model fine-tuned for classification
        self.intent_model_name = "microsoft/DialoGPT-medium"  # Good for dialogue understanding
        try:
            self.intent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            # For now, we'll use rule-based intent classification but keep model ready for fine-tuning
            logger.info("Intent classification: Using rule-based approach with BERT tokenizer ready")
        except Exception as e:
            logger.warning(f"Failed to load intent model: {e}")
            self.intent_tokenizer = None
        
        # 2. Entity Recognition Pipeline
        # Using a pre-trained NER model
        try:
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Named Entity Recognition model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
            self.ner_pipeline = None
        
        # 3. Conversational AI Model
        # Using a lightweight conversational model
        self.conversation_model_name = "microsoft/DialoGPT-small"  # Good balance of size and quality
        try:
            self.conversation_tokenizer = AutoTokenizer.from_pretrained(self.conversation_model_name)
            self.conversation_model = AutoModelForCausalLM.from_pretrained(
                self.conversation_model_name,
                quantization_config=quantization_config if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda":
                self.conversation_model.to(self.device)
            
            # Set pad token
            if self.conversation_tokenizer.pad_token is None:
                self.conversation_tokenizer.pad_token = self.conversation_tokenizer.eos_token
            
            logger.info("Conversational model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load conversational model: {e}")
            self.conversation_model = None
            self.conversation_tokenizer = None
        
        # 4. Text Generation Pipeline for Query Understanding
        try:
            self.text_generator = pipeline(
                "text-generation",
                model="distilgpt2",  # Lightweight but effective
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
            logger.info("Text generation pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load text generation pipeline: {e}")
            self.text_generator = None
    
    def _init_vector_store(self):
        """Initialize vector store for semantic search"""
        try:
            # Use sentence-transformers for embeddings (optimized for semantic search)
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Initialize PostgreSQL vector store
            self.db_connection_string = self._get_db_connection_string()
            
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.embedding_model = None
    
    def _get_db_connection_string(self) -> str:
        """Get PostgreSQL connection string for pgvector"""
        try:
            host = os.getenv('DB_HOST', 'localhost')
            port = int(os.getenv('DB_PORT', 5432))
            database = os.getenv('DB_NAME', 'brickbrain')
            user = os.getenv('DB_USER', 'brickbrain')
            password = os.getenv('DB_PASSWORD', 'brickbrain_password')
            
            return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        except Exception as e:
            logger.warning(f"Could not build DB connection string: {e}")
            return "postgresql+psycopg://brickbrain:brickbrain_password@localhost:5432/brickbrain"
    
    def _load_themes_from_database(self):
        """Load LEGO themes and build keyword mappings from database"""
        try:
            cursor = self.dbconn.cursor(cursor_factory=RealDictCursor)
            
            # Get all themes with their hierarchy
            cursor.execute("""
                SELECT id, name, parent_id 
                FROM themes 
                ORDER BY id
            """)
            themes_data = cursor.fetchall()
            
            # Build theme hierarchy and keyword mappings
            self._build_theme_mappings(themes_data)
            
            # Build enhanced interest categories based on actual theme data
            self._build_interest_categories(themes_data)
            
            self.theme_cache_timestamp = datetime.now()
            logger.info(f"Loaded {len(self.lego_themes)} themes from database")
            
        except Exception as e:
            logger.error(f"Failed to load themes from database: {e}")
            # Fall back to hardcoded themes
            self._load_fallback_themes()
    
    def _build_theme_mappings(self, themes_data: List[Dict]):
        """Build comprehensive theme keyword mappings"""
        self.lego_themes = {}
        self.theme_hierarchy = {}
        
        # Create theme hierarchy map
        theme_id_to_name = {}
        for theme in themes_data:
            theme_id_to_name[theme['id']] = theme['name']
            if theme['parent_id']:
                self.theme_hierarchy[theme['name']] = theme_id_to_name.get(theme['parent_id'])
        
        # Enhanced keyword mappings based on actual theme names
        theme_keywords = {}
        
        for theme in themes_data:
            theme_name = theme['name'].lower()
            keywords = []
            
            # Always include the exact theme name
            keywords.append(theme_name)
            keywords.append(theme_name.replace(' ', ''))  # Remove spaces
            keywords.append(theme_name.replace('-', ' '))  # Handle hyphens
            
            # Add specific keyword mappings for popular themes
            if 'star wars' in theme_name:
                keywords.extend(['starwars', 'jedi', 'sith', 'vader', 'millennium falcon', 
                               'luke skywalker', 'darth vader', 'empire', 'rebel', 'galaxy'])
            elif 'harry potter' in theme_name:
                keywords.extend(['hogwarts', 'wizarding', 'hermione', 'dumbledore', 'magic', 
                               'wizard', 'witch', 'quidditch', 'gryffindor', 'hogwarts castle'])
            elif 'technic' in theme_name:
                keywords.extend(['mechanical', 'gears', 'motors', 'pneumatic', 'motorized',
                               'engine', 'transmission', 'engineering'])
            elif 'city' in theme_name:
                keywords.extend(['police', 'fire department', 'ambulance', 'urban', 'metropolitan',
                               'rescue', 'emergency', 'hospital', 'construction site'])
            elif 'creator' in theme_name:
                keywords.extend(['3-in-1', 'modular', 'expert', 'alternative builds', 'multi-build'])
            elif 'friends' in theme_name:
                keywords.extend(['olivia', 'emma', 'mia', 'heartlake', 'friendship', 'girl'])
            elif 'ninjago' in theme_name:
                keywords.extend(['ninja', 'lloyd', 'kai', 'spinjitzu', 'sensei', 'dragon'])
            elif 'architecture' in theme_name:
                keywords.extend(['landmark', 'building', 'skyline', 'buildings', 'architectural',
                               'famous building', 'monument'])
            elif 'ideas' in theme_name or 'cuusoo' in theme_name:
                keywords.extend(['fan-designed', 'community', 'fan-created', 'crowdsourced'])
            elif 'castle' in theme_name:
                keywords.extend(['medieval', 'knight', 'dragon', 'fortress', 'kingdom', 'royal'])
            elif 'space' in theme_name:
                keywords.extend(['spaceship', 'galaxy', 'astronaut', 'rocket', 'sci-fi', 'alien',
                               'mars', 'planet', 'space station'])
            elif 'pirates' in theme_name:
                keywords.extend(['pirate ship', 'treasure', 'captain', 'caribbean', 'sailing'])
            elif 'train' in theme_name:
                keywords.extend(['locomotive', 'railway', 'railroad', 'station', 'cargo train'])
            elif 'car' in theme_name or 'racing' in theme_name or 'speed' in theme_name:
                keywords.extend(['vehicle', 'automobile', 'racing', 'speed', 'formula', 'sports car'])
            elif 'minecraft' in theme_name:
                keywords.extend(['block', 'creeper', 'steve', 'building game', 'pixelated'])
            elif 'disney' in theme_name:
                keywords.extend(['princess', 'mickey', 'mouse', 'fairy tale', 'animation'])
            elif 'super heroes' in theme_name or 'superhero' in theme_name:
                keywords.extend(['batman', 'superman', 'spider-man', 'avengers', 'marvel', 'dc'])
            elif 'jurassic' in theme_name:
                keywords.extend(['dinosaur', 'prehistoric', 't-rex', 'velociraptor', 'fossil'])
            elif any(word in theme_name for word in ['winter', 'christmas', 'holiday']):
                keywords.extend(['seasonal', 'festive', 'snow', 'santa', 'christmas tree'])
            elif 'modular' in theme_name:
                keywords.extend(['building', 'street', 'shop', 'cafe', 'detailed building'])
            
            # Add variations and common misspellings
            keywords.extend([
                theme_name.replace('lego', '').strip(),
                theme_name.replace('the', '').strip(),
                ''.join(theme_name.split())  # No spaces version
            ])
            
            # Remove empty strings and duplicates
            keywords = list(set([k for k in keywords if k and len(k) > 1]))
            theme_keywords[theme['name']] = keywords
        
        self.lego_themes = theme_keywords
    
    def _build_interest_categories(self, themes_data: List[Dict]):
        """Build enhanced interest categories based on theme data"""
        # Analyze themes to build comprehensive interest categories
        theme_names = [theme['name'].lower() for theme in themes_data]
        
        self.interest_categories = {
            'space': [
                'space', 'spaceship', 'galaxy', 'astronaut', 'rocket', 'sci-fi', 'alien',
                'mars', 'planet', 'space station', 'shuttle', 'satellite', 'cosmos'
            ],
            'vehicles': [
                'car', 'truck', 'vehicle', 'motorized', 'motor', 'driving', 'automobile',
                'racing', 'formula', 'sports car', 'motorcycle', 'bus', 'van', 'emergency vehicle'
            ],
            'buildings': [
                'building', 'house', 'castle', 'architecture', 'construction', 'skyscraper',
                'monument', 'landmark', 'tower', 'bridge', 'church', 'palace', 'fortress'
            ],
            'action_adventure': [
                'action', 'battle', 'fight', 'adventure', 'hero', 'superhero', 'combat',
                'mission', 'rescue', 'quest', 'exploration'
            ],
            'animals_nature': [
                'animal', 'pet', 'zoo', 'wildlife', 'creature', 'dinosaur', 'dragon',
                'forest', 'jungle', 'ocean', 'sea', 'nature', 'safari'
            ],
            'fantasy_magic': [
                'fantasy', 'magic', 'wizard', 'witch', 'dragon', 'fairy', 'unicorn',
                'magical', 'enchanted', 'mystical', 'legend', 'myth'
            ],
            'science_technology': [
                'robot', 'mechanical', 'engineering', 'technology', 'tech', 'cyberpunk',
                'futuristic', 'android', 'automation', 'programming'
            ],
            'history_culture': [
                'medieval', 'ancient', 'historical', 'cultural', 'traditional', 'viking',
                'roman', 'egyptian', 'samurai', 'knight', 'warrior'
            ],
            'trains_transport': [
                'train', 'railway', 'locomotive', 'cargo', 'passenger', 'subway',
                'monorail', 'tram', 'station', 'tracks'
            ],
            'emergency_services': [
                'police', 'fire', 'ambulance', 'rescue', 'hospital', 'emergency',
                'paramedic', 'firefighter', 'coast guard'
            ],
            'pirates_adventure': [
                'pirate', 'treasure', 'ship', 'sailing', 'ocean', 'island',
                'captain', 'crew', 'adventure', 'caribbean'
            ],
            'seasonal_holiday': [
                'christmas', 'holiday', 'winter', 'halloween', 'easter', 'valentine',
                'seasonal', 'festive', 'celebration', 'thanksgiving'
            ]
        }
        
        # Add theme-specific categories for major franchises found in data
        franchise_categories = {}
        for theme in themes_data:
            theme_name = theme['name'].lower()
            if 'star wars' in theme_name:
                franchise_categories['star_wars'] = [
                    'star wars', 'jedi', 'sith', 'force', 'empire', 'rebel',
                    'galaxy far far away', 'lightsaber', 'death star'
                ]
            elif 'harry potter' in theme_name:
                franchise_categories['harry_potter'] = [
                    'harry potter', 'wizarding world', 'hogwarts', 'magic', 'spell',
                    'quidditch', 'wizard', 'witch', 'magical creatures'
                ]
            elif 'marvel' in theme_name or 'superhero' in theme_name:
                franchise_categories['superheroes'] = [
                    'superhero', 'comic', 'marvel', 'dc', 'batman', 'superman',
                    'spider-man', 'avengers', 'justice league', 'powers'
                ]
        
        self.interest_categories.update(franchise_categories)
    
    def _load_fallback_themes(self):
        """Load fallback themes in case database loading fails"""
        logger.info("Loading fallback theme mappings")
        self.lego_themes = {
            'Star Wars': ['star wars', 'starwars', 'jedi', 'sith', 'vader', 'millennium falcon', 'space', 'spaceship', 'galaxy'],
            'Harry Potter': ['harry potter', 'hogwarts', 'wizarding', 'hermione', 'dumbledore'],
            'Technic': ['technic', 'mechanical', 'gears', 'motors', 'pneumatic', 'vehicles', 'cars', 'trucks', 'motorized'],
            'City': ['city', 'police', 'fire department', 'ambulance', 'train'],
            'Creator': ['creator', '3-in-1', 'modular', 'expert', 'vehicles'],
            'Friends': ['friends', 'olivia', 'emma', 'mia', 'heartlake'],
            'Ninjago': ['ninjago', 'ninja', 'lloyd', 'kai', 'spinjitzu'],
            'Architecture': ['architecture', 'landmark', 'building', 'skyline', 'buildings'],
            'Ideas': ['ideas', 'fan-designed', 'community'],
            'Castle': ['castle', 'medieval', 'knight', 'dragon', 'buildings']
        }
        
        self.interest_categories = {
            'space': ['space', 'spaceship', 'galaxy', 'astronaut', 'rocket', 'sci-fi'],
            'vehicles': ['car', 'truck', 'vehicle', 'motorized', 'motor', 'driving'],
            'buildings': ['building', 'house', 'castle', 'architecture', 'construction'],
            'action': ['action', 'battle', 'fight', 'adventure', 'hero'],
            'animals': ['animal', 'pet', 'zoo', 'wildlife', 'creature']
        }
    
    def refresh_themes_cache(self, force_refresh: bool = False):
        """Refresh themes from database if cache is stale"""
        if (force_refresh or 
            not self.theme_cache_timestamp or 
            (datetime.now() - self.theme_cache_timestamp).total_seconds() > 3600):  # 1 hour cache
            logger.info("Refreshing themes cache from database")
            self._load_themes_from_database()
    
    def fuzzy_match_theme(self, query_text: str, threshold: float = 0.7) -> List[str]:
        """
        Perform fuzzy matching on theme names for better recognition
        
        Args:
            query_text: Text to match against themes
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of matching theme names
        """
        try:
            from difflib import SequenceMatcher
        except ImportError:
            logger.warning("difflib not available for fuzzy matching")
            return []
        
        matches = []
        query_lower = query_text.lower()
        
        for theme_name, keywords in self.lego_themes.items():
            # Check exact theme name similarity
            similarity = SequenceMatcher(None, query_lower, theme_name.lower()).ratio()
            if similarity >= threshold:
                matches.append(theme_name)
                continue
            
            # Check similarity with each keyword
            for keyword in keywords:
                similarity = SequenceMatcher(None, query_lower, keyword.lower()).ratio()
                if similarity >= threshold:
                    matches.append(theme_name)
                    break
        
        return list(set(matches))
    
    def get_theme_hierarchy_context(self, theme_name: str) -> Dict[str, Any]:
        """
        Get hierarchical context for a theme (parent/child relationships)
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Dictionary with hierarchy information
        """
        context = {
            'theme': theme_name,
            'parent': self.theme_hierarchy.get(theme_name),
            'children': [],
            'siblings': []
        }
        
        # Find children
        for child, parent in self.theme_hierarchy.items():
            if parent == theme_name:
                context['children'].append(child)
            elif parent == context['parent'] and child != theme_name:
                context['siblings'].append(child)
        
        return context
    
    def enhance_theme_detection(self, query: str) -> Dict[str, Any]:
        """
        Enhanced theme detection combining exact matching, fuzzy matching, and hierarchy
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with detected themes and confidence scores
        """
        results = {
            'exact_matches': [],
            'fuzzy_matches': [],
            'hierarchy_suggestions': [],
            'confidence_scores': {}
        }
        
        query_lower = query.lower()
        
        # Exact keyword matching
        for theme_name, keywords in self.lego_themes.items():
            for keyword in keywords:
                if keyword in query_lower:
                    results['exact_matches'].append(theme_name)
                    results['confidence_scores'][theme_name] = 1.0
                    break
        
        # Fuzzy matching for themes not found exactly
        exact_match_names = set(results['exact_matches'])
        fuzzy_matches = self.fuzzy_match_theme(query, threshold=0.6)
        for match in fuzzy_matches:
            if match not in exact_match_names:
                results['fuzzy_matches'].append(match)
                results['confidence_scores'][match] = 0.7
        
        # Add hierarchy suggestions for exact matches
        for theme in results['exact_matches']:
            hierarchy = self.get_theme_hierarchy_context(theme)
            if hierarchy['children'] or hierarchy['siblings']:
                results['hierarchy_suggestions'].extend(hierarchy['children'][:3])  # Limit suggestions
                results['hierarchy_suggestions'].extend(hierarchy['siblings'][:2])
        
        # Remove duplicates while preserving order
        results['hierarchy_suggestions'] = list(dict.fromkeys(results['hierarchy_suggestions']))
        
        return results
    
    def validate_database_connection(self) -> bool:
        """Validate that database connection is working and themes table exists"""
        try:
            cursor = self.dbconn.cursor()
            cursor.execute("SELECT COUNT(*) FROM themes LIMIT 1")
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False
    
    def get_theme_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded themes for monitoring/debugging"""
        stats = {
            'total_themes': len(self.lego_themes),
            'total_keywords': sum(len(keywords) for keywords in self.lego_themes.values()),
            'interest_categories': len(self.interest_categories),
            'cache_age_minutes': 0,
            'database_connected': self.validate_database_connection()
        }
        
        if self.theme_cache_timestamp:
            cache_age = (datetime.now() - self.theme_cache_timestamp).total_seconds() / 60
            stats['cache_age_minutes'] = round(cache_age, 2)
        
        return stats
    
    def get_popular_themes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular themes based on set count from database
        
        Args:
            limit: Maximum number of themes to return
            
        Returns:
            List of theme dictionaries with popularity info
        """
        try:
            cursor = self.dbconn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT t.name, COUNT(s.set_num) as set_count
                FROM themes t
                LEFT JOIN sets s ON t.id = s.theme_id
                GROUP BY t.id, t.name
                HAVING COUNT(s.set_num) > 0
                ORDER BY set_count DESC
                LIMIT %s
            """, (limit,))
            
            popular_themes = []
            for row in cursor.fetchall():
                theme_info = {
                    'name': row['name'],
                    'set_count': row['set_count'],
                    'keywords': self.lego_themes.get(row['name'], [])
                }
                popular_themes.append(theme_info)
            
            return popular_themes
            
        except Exception as e:
            logger.error(f"Failed to get popular themes: {e}")
            return []
    
    def suggest_themes_for_user(self, user_query: str, user_age: Optional[int] = None, 
                               interest_categories: Optional[List[str]] = None) -> List[str]:
        """
        Suggest themes based on user query and profile
        
        Args:
            user_query: User's natural language query
            user_age: User's age (for age-appropriate suggestions)
            interest_categories: User's interest categories
            
        Returns:
            List of suggested theme names
        """
        suggestions = []
        
        # First, try enhanced theme detection
        detection_results = self.enhance_theme_detection(user_query)
        suggestions.extend(detection_results['exact_matches'])
        suggestions.extend(detection_results['fuzzy_matches'])
        suggestions.extend(detection_results['hierarchy_suggestions'])
        
        # Add suggestions based on interest categories
        if interest_categories:
            category_theme_mapping = {
                'space': ['Space', 'Star Wars', 'Galaxy Squad', 'Mars Mission'],
                'vehicles': ['Technic', 'City', 'Speed Champions', 'Racers'],
                'buildings': ['Architecture', 'Creator', 'Modular Buildings'],
                'action_adventure': ['Ninjago', 'Super Heroes DC', 'Super Heroes Marvel'],
                'fantasy_magic': ['Harry Potter', 'Castle', 'Elves'],
                'animals_nature': ['Friends', 'Duplo'],
            }
            
            for category in interest_categories:
                if category in category_theme_mapping:
                    suggestions.extend(category_theme_mapping[category])
        
        # Age-based suggestions
        if user_age:
            if user_age <= 5:
                suggestions.extend(['Duplo', 'Mickey & Friends'])
            elif user_age <= 12:
                suggestions.extend(['City', 'Friends', 'Ninjago', 'Creator'])
            elif user_age <= 16:
                suggestions.extend(['Technic', 'Architecture', 'Star Wars'])
            else:  # Adult
                suggestions.extend(['Creator Expert', 'Architecture', 'Icons', 'Modular Buildings'])
        
        # Remove duplicates while preserving order
        suggestions = list(dict.fromkeys(suggestions))
        
        # Filter to only themes that actually exist in our database
        valid_suggestions = [theme for theme in suggestions if theme in self.lego_themes]
        
        return valid_suggestions[:10]  # Return top 10 suggestions
    
    def classify_intent(self, query: str, conversation_context: str = "") -> str:
        """
        Classify the intent of a user query using HuggingFace models
        
        Args:
            query: User query string
            conversation_context: Previous conversation context
            
        Returns:
            Detected intent as string
        """
        query_lower = query.lower()
        
        # Enhanced rule-based classification with context awareness
        intent_scores = {}
        
        for intent, keywords in self.intents.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
                    # Boost score for exact matches
                    if keyword in query_lower.split():
                        score += 0.5
            
            # Context-based boosting
            if conversation_context:
                context_lower = conversation_context.lower()
                for keyword in keywords:
                    if keyword in context_lower:
                        score += 0.3  # Boost based on conversation context
            
            intent_scores[intent] = score
        
        # Find the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]
        
        # Default intent
        return 'search'
    
    def extract_entities_and_filters(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced entity and filter extraction from natural language query
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (entities_dict, filters_dict)
        """
        entities = {}
        filters = {}
        
        # Use NER model if available
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(query)
                for entity in ner_results:
                    if entity['entity_group'] == 'PER':
                        entities['recipient'] = entity['word']
                    elif entity['entity_group'] == 'ORG':
                        # Might be a LEGO theme
                        entities['potential_theme'] = entity['word']
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Rule-based extraction for LEGO-specific terms
        query_lower = query.lower()
        
        # Enhanced theme detection using database-loaded mappings
        theme_detection_results = self.enhance_theme_detection(query)
        
        # Combine exact and fuzzy matches
        detected_themes = theme_detection_results['exact_matches'] + theme_detection_results['fuzzy_matches']
        
        if detected_themes:
            filters['themes'] = detected_themes
            # Store confidence scores for potential ranking
            entities['theme_confidence'] = theme_detection_results['confidence_scores']
            
            # Add hierarchy suggestions for better recommendations
            if theme_detection_results['hierarchy_suggestions']:
                entities['related_themes'] = theme_detection_results['hierarchy_suggestions']
        
        # Extract interest categories using enhanced mappings
        categories_to_check = self.interest_categories if hasattr(self, 'interest_categories') and self.interest_categories else {
            'space': ['space', 'spaceship', 'galaxy', 'astronaut', 'rocket', 'sci-fi'],
            'vehicles': ['car', 'truck', 'vehicle', 'motorized', 'motor', 'driving'],
            'buildings': ['building', 'house', 'castle', 'architecture', 'construction'],
            'action': ['action', 'battle', 'fight', 'adventure', 'hero'],
            'animals': ['animal', 'pet', 'zoo', 'wildlife', 'creature']
        }
        
        detected_categories = []
        category_confidence = {}
        for category, keywords in categories_to_check.items():
            category_score = 0
            matching_keywords = []
            for keyword in keywords:
                if keyword in query_lower:
                    category_score += 1
                    matching_keywords.append(keyword)
            
            if category_score > 0:
                detected_categories.append(category)
                category_confidence[category] = category_score / len(keywords)  # Normalized confidence
                entities[f'{category}_keywords'] = matching_keywords
        
        if detected_categories:
            entities['interest_categories'] = detected_categories
            entities['category_confidence'] = category_confidence
        
        # Extract piece count with ranges
        piece_patterns = [
            r'(\d+)\s*(?:to|[-–])\s*(\d+)\s*(?:pieces?|parts?)',  # Range like "1000 to 2000 pieces"
            r'between\s*(\d+)\s*and\s*(\d+)\s*(?:pieces?|parts?)',  # "between 1000 and 2000 pieces"
            r'over\s*(\d+)\s*(?:pieces?|parts?)',  # "over 1000 pieces"
            r'under\s*(\d+)\s*(?:pieces?|parts?)',  # "under 500 pieces"
            r'(\d+)\s*(?:pieces?|parts?)'  # Just "500 pieces"
        ]
        
        for i, pattern in enumerate(piece_patterns):
            piece_match = re.search(pattern, query_lower)
            if piece_match:
                if i == 0 or i == 1:  # Range patterns
                    min_pieces = int(piece_match.group(1))
                    max_pieces = int(piece_match.group(2))
                    filters['min_pieces'] = min_pieces
                    filters['max_pieces'] = max_pieces
                elif i == 2:  # "over X pieces"
                    min_pieces = int(piece_match.group(1))
                    filters['min_pieces'] = min_pieces - 100  # Allow some flexibility
                    filters['max_pieces'] = min_pieces + 1000  # Upper bound
                elif i == 3:  # "under X pieces"
                    max_pieces = int(piece_match.group(1))
                    filters['min_pieces'] = max(1, max_pieces - 400)  # Lower bound
                    filters['max_pieces'] = max_pieces + 100  # Allow some flexibility
                else:  # Single number
                    piece_count = int(piece_match.group(1))
                    # Interpret as target piece count with flexibility
                    filters['min_pieces'] = max(1, piece_count - 100)
                    filters['max_pieces'] = piece_count + 100
                break
        
        # Extract age information
        age_patterns = [
            r'(\d+)[\s-]*(?:year[\s-]*old|yo|years?)',
            r'age[\s]*(\d+)',
            r'for[\s]*(\d+)[\s]*year',
            r'(\d+)(?:st|nd|rd|th)[\s]*birthday',  # "6th birthday"
            r'(\d+)[\s]*(?:st|nd|rd|th)[\s]*(?:birthday|bday)'  # "6 th birthday"
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, query_lower)
            if age_match:
                age = int(age_match.group(1))
                filters['min_age'] = max(1, age - 2)
                filters['max_age'] = age + 5
                entities['age'] = age
                break
        
        # Extract recipient information
        recipient_patterns = {
            'nephew': ['nephew'],
            'niece': ['niece'],
            'son': ['son', 'boy'],
            'daughter': ['daughter', 'girl'],
            'child': ['child', 'kid'],
            'adult': ['adult', 'grown-up', 'myself', 'me'],
            'teenager': ['teenager', 'teen']
        }
        
        for recipient, keywords in recipient_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['recipient'] = recipient
                break
        
        # Extract occasion
        occasions = ['birthday', 'christmas', 'holiday', 'gift', 'present']
        for occasion in occasions:
            if occasion in query_lower:
                entities['occasion'] = occasion
                break
        
        # Extract experience level
        experience_levels = {
            'beginner': ['beginner', 'starter', 'new', 'first time', 'easy'],
            'intermediate': ['intermediate', 'moderate', 'some experience'],
            'expert': ['expert', 'advanced', 'experienced', 'master', 'professional']
        }
        
        for level, keywords in experience_levels.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['experience_level'] = level
                break
        
        # Extract building preferences
        building_preferences = {
            'challenging': ['challenging', 'difficult', 'complex'],
            'detailed': ['detailed', 'intricate', 'precise'],
            'quick_build': ['quick', 'fast', 'simple', 'weekend']
        }
        
        for preference, keywords in building_preferences.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['building_preference'] = preference
                break
        
        # Extract special features
        special_features = []
        feature_keywords = {
            'minifigures': ['minifigure', 'minifig', 'figure', 'character'],
            'motorized': ['motorized', 'motor', 'powered', 'electric'],
            'lights': ['light', 'led', 'illuminated', 'glowing'],
            'sound': ['sound', 'music', 'audio'],
            'remote_control': ['remote control', 'rc', 'controllable']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                special_features.append(feature)
        
        if special_features:
            entities['special_features'] = special_features
        
        # Extract time constraints
        time_constraints = {
            'weekend_project': ['weekend', 'saturday', 'sunday', 'quick build'],
            'vacation_project': ['vacation', 'holiday break', 'long project'],
            'daily_build': ['daily', 'bit by bit', 'gradually']
        }
        
        for constraint, keywords in time_constraints.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['time_constraint'] = constraint
                break
        
        # Extract budget/price information
        price_patterns = [
            r'under[\s]*\$?(\d+)',
            r'less than[\s]*\$?(\d+)',
            r'budget[\s]*\$?(\d+)',
            r'\$(\d+)[\s]*(?:or less|max|maximum)'
        ]
        
        for pattern in price_patterns:
            price_match = re.search(pattern, query_lower)
            if price_match:
                max_price = float(price_match.group(1))
                filters['price_range'] = [0, max_price]
                entities['budget'] = max_price
                break
        
        # Extract complexity level (enhanced)
        if any(word in query_lower for word in ['complex', 'challenging', 'detailed', 'advanced', 'expert', 'difficult']):
            entities['complexity'] = 'advanced'
        elif any(word in query_lower for word in ['easy', 'simple', 'beginner', 'starter', 'basic']):
            entities['complexity'] = 'beginner'
        else:
            entities['complexity'] = 'intermediate'
        
        return entities, filters
    
    def generate_conversational_response(self, query: str, context: str = "", 
                                       recommendations: List[Dict] = None) -> str:
        """
        Generate a conversational response using HuggingFace models
        
        Args:
            query: User query
            context: Conversation context
            recommendations: List of LEGO set recommendations
            
        Returns:
            Generated response string
        """
        # Create a prompt for response generation
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        prompt_parts.append(f"User: {query}")
        
        if recommendations:
            prompt_parts.append("Recommendations available:")
            for i, rec in enumerate(recommendations[:3], 1):
                prompt_parts.append(f"{i}. {rec.get('name', 'Unknown')} - {rec.get('theme', 'N/A')}")
        
        prompt = "\n".join(prompt_parts)
        prompt += "\nAssistant: I'd be happy to help you find the perfect LEGO set! "
        
        # Use text generation model if available
        if self.text_generator:
            try:
                # Generate response
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                generated_text = response[0]['generated_text']
                # Extract only the assistant's response
                assistant_response = generated_text.split("Assistant:")[-1].strip()
                
                # Clean up the response
                assistant_response = assistant_response.split('\n')[0]  # Take first line
                assistant_response = assistant_response[:200]  # Limit length
                
                return assistant_response
            
            except Exception as e:
                logger.warning(f"Text generation failed: {e}")
        
        # Fallback to template-based responses
        return self._generate_template_response(query, recommendations)
    
    def _generate_template_response(self, query: str, recommendations: List[Dict] = None) -> str:
        """Generate template-based responses as fallback"""
        query_lower = query.lower()
        
        response_templates = {
            'gift': "I found some great LEGO sets that would make perfect gifts! ",
            'star wars': "Here are some amazing Star Wars LEGO sets! ",
            'beginner': "These sets are perfect for beginners! ",
            'advanced': "Here are some challenging sets for experienced builders! ",
            'budget': "I found some affordable options within your budget! ",
            'similar': "Based on your interests, here are some similar sets! ",
        }
        
        # Find appropriate template
        response = "Here are some LEGO sets I think you'll love! "
        for keyword, template in response_templates.items():
            if keyword in query_lower:
                response = template
                break
        
        # Add recommendation summary
        if recommendations:
            themes = list(set(rec.get('theme', 'N/A') for rec in recommendations[:3]))
            if themes and themes[0] != 'N/A':
                response += f"I've found sets from {', '.join(themes)} themes. "
        
        return response
    
    def process_natural_language_query(self, query: str, user_id: Optional[int] = None,
                                     use_conversation_context: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query and return structured results
        
        Args:
            query: Natural language query
            user_id: Optional user ID for personalization
            use_conversation_context: Whether to use conversation context
            
        Returns:
            Dictionary with processed query results
        """
        # Get conversation context
        conversation_context = ""
        if use_conversation_context:
            conversation_context = self.conversation_memory.get_recent_context()
        
        # Classify intent
        intent = self.classify_intent(query, conversation_context)
        
        # Extract entities and filters
        entities, filters = self.extract_entities_and_filters(query)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(query, intent, entities, filters)
        
        # Create semantic query for vector search
        semantic_query = self._create_semantic_query(query, entities, filters)
        
        result = {
            'query': query,
            'intent': intent,
            'entities': entities,
            'filters': filters,
            'semantic_query': semantic_query,
            'confidence': confidence,
            'conversation_context': conversation_context
        }
        
        return result
    
    def _calculate_confidence(self, query: str, intent: str, entities: Dict, filters: Dict) -> float:
        """Calculate confidence score for the query processing"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on extracted information
        if entities:
            confidence += len(entities) * 0.1
        
        if filters:
            confidence += len(filters) * 0.15
        
        # Boost confidence for clear intent indicators
        if intent != 'search':  # More specific intent
            confidence += 0.2
        
        # Boost confidence for longer, more descriptive queries
        if len(query.split()) > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _create_semantic_query(self, query: str, entities: Dict, filters: Dict) -> str:
        """Create an enhanced semantic query for vector search"""
        semantic_parts = [query]
        
        # Add entity information
        if entities.get('recipient'):
            semantic_parts.append(f"for {entities['recipient']}")
        
        if entities.get('occasion'):
            semantic_parts.append(f"{entities['occasion']} gift")
        
        if entities.get('complexity'):
            semantic_parts.append(f"{entities['complexity']} difficulty")
        
        # Add filter information
        if filters.get('themes'):
            semantic_parts.extend(filters['themes'])
        
        return " ".join(semantic_parts)
    
    def search_recommendations(self, processed_query: Dict[str, Any], 
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for LEGO set recommendations based on processed query
        
        Args:
            processed_query: Result from process_natural_language_query
            top_k: Number of recommendations to return
            
        Returns:
            List of LEGO set recommendations
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available, returning empty results")
            return []

        try:
            # Check if we have theme filters to prioritize
            filters = processed_query.get('filters', {})
            themes = filters.get('themes', [])
            
            if themes:
                # Search specifically for sets in the requested themes first
                results = self._query_by_themes(themes, top_k)
                if results:
                    # Apply other filters to theme-based results
                    filtered_results = self._apply_filters(results, filters)
                    for result in filtered_results:
                        result['confidence'] = processed_query['confidence']
                        result['intent'] = processed_query['intent']
                        result['relevance_score'] = 0.9  # Higher relevance for theme matches
                    return filtered_results[:top_k]
            
            # Fallback to general search if no theme-specific results
            semantic_query = processed_query['semantic_query']
            query_embedding = self.embedding_model.encode([semantic_query])
            
            # Query the vector database
            results = self._query_vector_database(query_embedding[0], top_k)
            
            # Apply filters
            filtered_results = self._apply_filters(results, filters)
            
            # Add confidence and intent to results
            for result in filtered_results:
                result['confidence'] = processed_query['confidence']
                result['intent'] = processed_query['intent']
            
            return filtered_results[:top_k]
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _query_by_themes(self, themes: List[str], top_k: int) -> List[Dict]:
        """Query sets specifically by theme names"""
        try:
            cursor = self.dbconn.cursor(cursor_factory=RealDictCursor)
            
            # Create a case-insensitive theme query
            theme_conditions = []
            params = []
            for theme in themes:
                theme_conditions.append("LOWER(t.name) LIKE LOWER(%s)")
                params.append(f"%{theme}%")
            
            query = f"""
                SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url,
                       t.name as theme_name, t.id as theme_id
                FROM sets s
                LEFT JOIN themes t ON s.theme_id = t.id
                WHERE s.num_parts IS NOT NULL 
                AND s.num_parts > 50
                AND s.year >= 2000
                AND ({' OR '.join(theme_conditions)})
                ORDER BY s.num_parts DESC, s.year DESC
                LIMIT %s
            """
            
            params.append(top_k * 2)  # Get more for variety
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'set_num': row['set_num'],
                    'name': row['name'], 
                    'year': row['year'],
                    'num_parts': row['num_parts'],
                    'theme': row['theme_name'] or 'Generic',
                    'theme_id': row['theme_id'],
                    'img_url': row['img_url'],
                    'relevance_score': 0.9  # High relevance for theme matches
                }
                results.append(result)
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Theme query failed: {e}")
            return []

    def _query_vector_database(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Query the vector database for similar LEGO sets"""
        try:
            # For now, perform a simple database query to get real LEGO sets
            # TODO: Implement semantic vector search with embeddings
            cursor = self.dbconn.cursor(cursor_factory=RealDictCursor)
            
            # Get random popular sets as a basic implementation
            cursor.execute("""
                SELECT s.set_num, s.name, s.year, s.num_parts, s.img_url,
                       t.name as theme_name, t.id as theme_id
                FROM sets s
                LEFT JOIN themes t ON s.theme_id = t.id
                WHERE s.num_parts IS NOT NULL 
                AND s.num_parts > 50
                AND s.year >= 2005
                ORDER BY RANDOM()
                LIMIT %s
            """, (top_k * 3,))  # Get more than needed for variety
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'set_num': row['set_num'],
                    'name': row['name'],
                    'year': row['year'],
                    'num_parts': row['num_parts'],
                    'theme': row['theme_name'] or 'Generic',
                    'theme_id': row['theme_id'],
                    'img_url': row['img_url'],
                    'relevance_score': 0.8  # Default relevance for now
                }
                results.append(result)
            
            cursor.close()
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply filters to search results"""
        if not filters:
            return results
        
        filtered = []
        for result in results:
            include = True
            
            # Theme filter
            if filters.get('themes') and result.get('theme'):
                if result['theme'] not in filters['themes']:
                    include = False
            
            # Piece count filter
            if filters.get('min_pieces') and result.get('num_parts'):
                if result['num_parts'] < filters['min_pieces']:
                    include = False
            
            if filters.get('max_pieces') and result.get('num_parts'):
                if result['num_parts'] > filters['max_pieces']:
                    include = False
            
            # Age filter
            if filters.get('min_age') and result.get('min_age'):
                if result['min_age'] > filters['min_age']:
                    include = False
            
            if filters.get('max_age') and result.get('max_age'):
                if result['max_age'] < filters['max_age']:
                    include = False
            
            if include:
                filtered.append(result)
        
        return filtered
    
    def add_conversation_interaction(self, user_message: str, assistant_response: str,
                                   recommendations: List[Dict] = None):
        """Add a conversation interaction to memory"""
        context = {}
        if recommendations:
            context['recommendation_count'] = len(recommendations)
            context['themes'] = list(set(r.get('theme', '') for r in recommendations))
        
        self.conversation_memory.add_interaction(user_message, assistant_response, context)
        
        # Update user context
        self.user_context['previous_searches'].append({
            'query': user_message,
            'timestamp': datetime.now().isoformat(),
            'response_summary': assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response
        })
        
        # Keep only recent searches
        if len(self.user_context['previous_searches']) > 20:
            self.user_context['previous_searches'] = self.user_context['previous_searches'][-20:]
    
    def get_conversation_context(self) -> ConversationContext:
        """Get current conversation context"""
        recent_queries = [search['query'] for search in self.user_context['previous_searches'][-5:]]
        conversation_summary = self.conversation_memory.get_conversation_summary()
        
        return ConversationContext(
            user_preferences=self.user_context.get('preferences', {}),
            previous_recommendations=self.user_context.get('previous_recommendations', []),
            current_session_queries=recent_queries,
            follow_up_context={},
            conversation_summary=conversation_summary
        )
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences based on feedback"""
        if 'preferences' not in self.user_context:
            self.user_context['preferences'] = {}
        
        self.user_context['preferences'].update(preferences)
    
    def record_user_feedback(self, set_num: str, feedback: str, rating: Optional[int] = None):
        """Record user feedback for learning"""
        feedback_entry = {
            'set_num': set_num,
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        if feedback == 'liked':
            self.user_context['liked_sets'].append(feedback_entry)
        elif feedback == 'disliked':
            self.user_context['disliked_sets'].append(feedback_entry)
        
        # Learn from feedback
        self._learn_from_feedback(set_num, feedback)
    
    def _learn_from_feedback(self, set_num: str, feedback: str):
        """Learn user preferences from feedback"""
        # Query database for set information
        try:
            cursor = self.dbconn.cursor()
            cursor.execute("""
                SELECT theme_name, num_parts, min_age 
                FROM sets 
                WHERE set_num = %s
            """, (set_num,))
            
            result = cursor.fetchone()
            if result:
                theme, num_parts, min_age = result
                
                # Update theme preferences
                if 'themes' not in self.user_context['preferences']:
                    self.user_context['preferences']['themes'] = {}
                
                if feedback == 'liked':
                    self.user_context['preferences']['themes'][theme] = \
                        self.user_context['preferences']['themes'].get(theme, 0) + 1
                elif feedback == 'disliked':
                    self.user_context['preferences']['themes'][theme] = \
                        self.user_context['preferences']['themes'].get(theme, 0) - 1
            
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
    
    def clear_conversation_memory(self):
        """Clear conversation memory and reset user context"""
        self.conversation_memory.clear()
        self.user_context = {
            'preferences': {},
            'previous_searches': [],
            'liked_sets': [],
            'disliked_sets': [],
            'conversation_session': datetime.now().isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the HuggingFace NLP system"""
        status = {
            'nlp_system': 'healthy',
            'models_loaded': {
                'embedding_model': self.embedding_model is not None,
                'conversation_model': self.conversation_model is not None,
                'ner_pipeline': self.ner_pipeline is not None,
                'text_generator': self.text_generator is not None
            },
            'device': self.device,
            'quantization_enabled': self.use_quantization,
            'conversation_memory': len(self.conversation_memory.conversations),
            'user_preferences': len(self.user_context.get('preferences', {}))
        }
        
        # Check if at least basic functionality is available
        if not any(status['models_loaded'].values()):
            status['nlp_system'] = 'degraded'
        
        return status
