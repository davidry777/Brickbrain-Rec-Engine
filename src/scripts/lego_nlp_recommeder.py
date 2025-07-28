import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# LangChain imports (updated to avoid deprecation warnings)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_postgres import PGVector
from langchain_core.documents import Document

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Simple conversation memory replacement to avoid deprecated LangChain memory
class SimpleConversationMemory:
    """Simple conversation memory that maintains chat history."""
    
    def __init__(self, memory_key="chat_history", return_messages=True, input_key="human_input", output_key="ai_response"):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        self.chat_history = []
    
    def save_context(self, inputs, outputs):
        """Save context to memory."""
        human_input = inputs.get(self.input_key, "")
        ai_response = outputs.get(self.output_key, "")
        
        if self.return_messages:
            self.chat_history.append(HumanMessage(content=human_input))
            self.chat_history.append(AIMessage(content=ai_response))
        else:
            self.chat_history.append({"human": human_input, "ai": ai_response})
    
    def load_memory_variables(self, inputs):
        """Load memory variables."""
        return {self.memory_key: self.chat_history}
    
    def clear(self):
        """Clear the memory."""
        self.chat_history = []

# For local embeddings (no API required)
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama as OllamaLLM

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
    themes: Optional[List[str]] = Field(None, description="LEGO themes like 'Star Wars', 'City', 'Technic'")
    min_pieces: Optional[int] = Field(None, description="Minimum number of pieces")
    max_pieces: Optional[int] = Field(None, description="Maximum number of pieces")
    min_age: Optional[int] = Field(None, description="Minimum age recommendation")
    max_age: Optional[int] = Field(None, description="Maximum age recommendation")
    complexity: Optional[str] = Field(None, description="Complexity level: 'simple', 'moderate', 'complex'")
    budget_min: Optional[float] = Field(None, description="Minimum price in USD")
    budget_max: Optional[float] = Field(None, description="Maximum price in USD")
    building_time: Optional[str] = Field(None, description="Estimated building time")
    special_features: Optional[List[str]] = Field(None, description="Special features like 'motorized', 'light-up', 'minifigures'")

class NLPRecommender:
    """Natural Language Processing for LEGO recommendations using LangChain"""
    def __init__(self, dbcon, use_openai: bool = False):
        self.dbconn = dbcon
        self.use_openai = use_openai
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Initialize to None first
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.docs = []
        self.intents = {
            'search': ['find', 'search', 'looking for', 'show me', 'want', 'need'],
            'recommend_similar': ['similar to', 'like', 'comparable', 'alternative to'],
            'gift_recommendation': ['gift', 'present', 'birthday', 'christmas', 'for my nephew', 'for my son', 'for my daughter', 'for a child'],
            'collection_advice': ['should i buy', 'worth it', 'good investment', 'collection', 'for my collection']
        }
        self.is_initialized = False

        # Initialize conversation memory
        self.conversation_memory = SimpleConversationMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="human_input",
            output_key="ai_response"
        )
        
        # User context storage for personalized recommendations
        self.user_context = {
            'preferences': {},
            'previous_searches': [],
            'liked_sets': [],
            'disliked_sets': [],
            'conversation_session': datetime.now().isoformat()
        }

        # Get database connection parameters for pgvector
        self.db_connection_string = self._get_db_connection_string()

        # Initialize embeddings and vector store
        if self.use_openai:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        else:
            # Use free Hugging Face model for embeddings
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                # Fallback to old import if new package not available
                from langchain_community.embeddings import HuggingFaceEmbeddings
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": self.device
                },
                encode_kwargs={"normalize_embeddings": True}
            )
            
            try:
                from langchain_ollama import OllamaLLM
                self.llm = OllamaLLM(model="mistral")  # Use Ollama as LLM
            except ImportError:
                try:
                    # Fallback to old import if new package not available
                    from langchain_community.llms import Ollama as OllamaLLM
                    self.llm = OllamaLLM(model="mistral")  # Use Ollama as LLM
                except ImportError:
                    logger.warning("Ollama not available, using text-based processing only")
                    self.llm = None
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}")
                logger.info("Falling back to text-based processing")
                self.llm = None
                    
            if self.llm is None:
                logger.info("Running in text-only mode without LLM")
            else:
                logger.info("LLM initialized successfully")

        # Initialize PGVector store (will be created when prep_vectorDB is called)
        self.vectorstore = None

        self.is_initialized = True

    def _get_db_connection_string(self) -> str:
        """Get PostgreSQL connection string for pgvector (psycopg3 format)."""
        try:
            # Get connection parameters from environment
            import os
            host = os.getenv('DB_HOST', 'localhost')
            port = int(os.getenv('DB_PORT', 5432))
            database = os.getenv('DB_NAME', 'brickbrain')
            user = os.getenv('DB_USER', 'brickbrain')
            password = os.getenv('DB_PASSWORD', 'brickbrain_password')
            
            # Use psycopg3 format (note: driver name is 'psycopg' not 'psycopg3')
            connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
            return connection_string
        except Exception as e:
            logger.warning(f"Could not build DB connection string: {e}")
            # Fallback to default connection string
            return "postgresql+psycopg://brickbrain:brickbrain_password@localhost:5432/brickbrain"

    def prep_vectorDB(self, limit_sets: Optional[int] = None, use_cloud: bool = False):
        """
        Prepare vector database with LEGO set descriptions and metadata
        
        :param limit_sets: Optional limit on number of sets to process (for faster setup)
        :param use_cloud: If True, prepare data for cloud processing instead of local
        """
        logger.info("Preparing vector database...")

        # Load LEGO set data with rich descriptions
        query = """
        SELECT 
            s.set_num,
            s.name,
            s.year,
            s.num_parts,
            t.name as theme_name,
            pt.name as parent_theme_name,
            COUNT(DISTINCT ip.color_id) as num_colors,
            COUNT(DISTINCT im.fig_num) as num_minifigs,
            STRING_AGG(DISTINCT cat.name, ', ') as part_categories
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN themes pt ON t.parent_id = pt.id
        LEFT JOIN inventories i ON s.set_num = i.set_num
        LEFT JOIN inventory_parts ip ON i.id = ip.inventory_id
        LEFT JOIN inventory_minifigs im ON i.id = im.inventory_id
        LEFT JOIN parts p ON ip.part_num = p.part_num
        LEFT JOIN part_categories cat ON p.part_cat_id = cat.id
        WHERE s.num_parts > 0
        GROUP BY s.set_num, s.name, s.year, s.num_parts, t.name, pt.name
        ORDER BY s.num_parts DESC, s.year DESC
        """
        
        if limit_sets:
            query += f" LIMIT {limit_sets}"

        df = pd.read_sql_query(query, self.dbconn)
        
        if df.empty:
            logger.warning("No LEGO sets found in database")
            return

        # Create documents with rich descriptions
        docs = []
        for _, row in df.iterrows():
            description = self._create_set_description(row)

            # Create metadata
            metadata = {
                'set_num': row['set_num'],
                'name': row['name'],
                'year': int(row['year']),
                'num_parts': int(row['num_parts']),
                'theme': row['theme_name'],
                'parent_theme': row['parent_theme_name'],
                'num_colors': int(row['num_colors']) if pd.notna(row['num_colors']) else 0,
                'num_minifigs': int(row['num_minifigs']) if pd.notna(row['num_minifigs']) else 0,
                'complexity': self._estimate_complexity(row)
            }
            docs.append(Document(page_content=description, metadata=metadata))

        self.docs = docs
        logger.info(f"Created {len(docs)} documents for vector database")

        if use_cloud:
            # For cloud processing, save documents and return
            self._prepare_for_cloud_processing()
            return
        
        # Local processing
        self._create_local_vectorstore()
    
    def _prepare_for_cloud_processing(self):
        """Prepare documents for cloud-based vector processing"""
        import json
        
        # Save documents in a format suitable for cloud processing
        cloud_data = []
        for doc in self.docs:
            cloud_data.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        # Save to file for cloud upload
        with open('embeddings/lego_sets_for_cloud.json', 'w') as f:
            json.dump(cloud_data, f)
        
        logger.info(f"Prepared {len(cloud_data)} documents for cloud processing")
        logger.info("Upload 'embeddings/lego_sets_for_cloud.json' to your cloud service")
    
    def _create_local_vectorstore(self):
        """Create vector store using LangChain's PGVector."""
        logger.info(f"Creating PGVector store with {len(self.docs)} documents...")
        
        # Create PGVector store with documents
        collection_name = "lego_sets"
        
        try:
            self.vectorstore = PGVector.from_documents(
                documents=self.docs,
                embedding=self.embeddings,
                collection_name=collection_name,
                connection=self.db_connection_string,
                use_jsonb=True,  # Use JSONB for metadata storage
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
            
            logger.info(f"Successfully created PGVector store with {len(self.docs)} documents")
            
        except Exception as e:
            logger.error(f"Failed to create PGVector store: {e}")
            logger.info("Falling back to Chroma vector store...")
            
            # Fallback to Chroma if PGVector fails
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=self.docs,
                    embedding=self.embeddings,
                    persist_directory="./chroma_db"
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
                logger.info(f"Successfully created Chroma fallback store with {len(self.docs)} documents")
            except Exception as fallback_error:
                logger.error(f"Fallback to Chroma also failed: {fallback_error}")
                raise
    
    def load_cloud_embeddings(self, embeddings_path: str, metadata_path: str):
        """
        Load pre-computed embeddings from cloud processing into PGVector
        
        :param embeddings_path: Path to numpy array of embeddings (deprecated for PGVector)
        :param metadata_path: Path to JSON file with metadata
        """
        import json
        
        logger.info("Loading cloud-processed documents into PGVector...")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        # Create documents from metadata
        documents = []
        for meta in metadata_list:
            doc = Document(
                page_content=meta['content'],
                metadata=meta['metadata']
            )
            documents.append(doc)
        
        # Create PGVector store (embeddings will be computed automatically)
        self.docs = documents
        collection_name = "lego_sets_cloud"
        
        try:
            self.vectorstore = PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                connection=self.db_connection_string,
                use_jsonb=True,
            )
            
            # Set up retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
            
            logger.info(f"Successfully loaded {len(documents)} cloud-processed documents into PGVector")
            
        except Exception as e:
            logger.error(f"Failed to load into PGVector: {e}")
            raise
    
    def _create_set_description(self, row) -> str:
        """
        Create a rich description for a LEGO set based on its metadata.
        This can be extended with more complex logic or external data sources.

        :param row: DataFrame row containing LEGO set metadata
        :return: Formatted description string
        """
        desc_parts = [
            f"LEGO {row['name']} (Set {row['set_num']})",
            f"from the {row['theme_name']} theme"
        ]
        
        if row['parent_theme_name'] and row['parent_theme_name'] != row['theme_name']:
            desc_parts.append(f"part of the {row['parent_theme_name']} collection")
        
        desc_parts.extend([
            f"released in {row['year']}",
            f"with {row['num_parts']} pieces"
        ])
        
        if row['num_colors'] > 0:
            desc_parts.append(f"featuring {row['num_colors']} different colors")
        
        if row['num_minifigs'] > 0:
            desc_parts.append(f"includes {row['num_minifigs']} minifigures")
        
        if row['part_categories']:
            desc_parts.append(f"contains parts from categories: {row['part_categories']}")
        
        # Add complexity description
        complexity = self._estimate_complexity(row)
        if complexity == 'simple':
            desc_parts.append("suitable for beginners with straightforward building")
        elif complexity == 'complex':
            desc_parts.append("challenging build for experienced builders")
        else:
            desc_parts.append("moderate complexity suitable for most builders")
        
        return ". ".join(desc_parts)
    
    def _estimate_complexity(self, row) -> str:
        """
        Estimate the complexity of a LEGO set based on its metadata.

        :param row: DataFrame row containing LEGO set metadata
        :return: Complexity level as 'simple', 'moderate', or 'complex'
        """
        colors = row['num_colors'] if pd.notna(row['num_colors']) else 0

        if row['num_parts'] < 100 or colors < 5:
            return 'simple'
        elif row['num_parts'] > 1000 or colors > 20:
            return 'complex'
        else:
            return 'moderate'
        
    def _calculate_confidence(self, query: str, intent: str, filters: Dict[str, Any], entities: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on query processing results.
        
        :param query: Original natural language query
        :param intent: Detected intent
        :param filters: Extracted filters
        :param entities: Extracted entities
        :return: Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # Intent confidence (0.2 - 0.4)
        intent_keywords = self.intents.get(intent, [])
        intent_matches = sum(1 for keyword in intent_keywords if keyword in query.lower())
        if intent_matches > 0:
            intent_confidence = min(0.4, 0.2 + (intent_matches * 0.1))
        else:
            intent_confidence = 0.2  # Default intent fallback
        confidence_factors.append(intent_confidence)
        
        # Filter extraction confidence (0.0 - 0.3)
        filter_confidence = 0.0
        if filters:
            # Award points for different types of filters
            filter_types = len(filters)
            specific_filters = ['themes', 'min_pieces', 'max_pieces', 'complexity']
            specific_matches = sum(1 for f in specific_filters if f in filters)
            filter_confidence = min(0.3, (filter_types * 0.05) + (specific_matches * 0.08))
        confidence_factors.append(filter_confidence)
        
        # Entity extraction confidence (0.0 - 0.2)
        entity_confidence = 0.0
        if entities:
            entity_types = len(entities)
            # Give extra weight to key entities that improve recommendations
            key_entities = ['recipient', 'age', 'occasion', 'experience_level']
            key_entity_count = sum(1 for key in key_entities if key in entities)
            entity_confidence = min(0.2, (entity_types * 0.03) + (key_entity_count * 0.05))
        confidence_factors.append(entity_confidence)
        
        # Query clarity confidence (0.0 - 0.1)
        query_words = len(query.split())
        if query_words >= 3:  # Reasonable query length
            clarity_confidence = min(0.1, 0.05 + (min(query_words, 10) * 0.005))
        else:
            clarity_confidence = 0.02  # Very short queries are less clear
        confidence_factors.append(clarity_confidence)
        
        # Calculate total confidence
        total_confidence = sum(confidence_factors)
        
        # Ensure confidence is within bounds
        return max(0.1, min(1.0, total_confidence))

    def process_nl_query(self, query: str, user_context: Optional[Dict]) -> NLQueryResult:
        """
        Process a natural language query to extract intent, filters, and semantic query.

        :param query: Natural language query string
        :param user_context: Optional user context for personalized recommendations
        :return: NLQueryResult containing intent, filters, confidence, entities, and semantic query
        """
        # Detect Intent
        intent = self._detect_intent(query)

        # Extract filters and entities
        if hasattr(self, 'llm') and self.llm:
            filters = self._extract_filters_llm(query)
            entities = self._extract_entities_llm(query)
        else:
            filters = self._extract_filters_regex(query)
            entities = self._extract_entities_regex(query)

        # Create semantic query for embediing search
        semantic_query = self._create_semantic_query(query, filters, entities)

        # Calculate dynamic confidence
        confidence = self._calculate_confidence(query, intent, filters, entities)

        return NLQueryResult(
            intent=intent,
            filters=filters,
            confidence=confidence,
            extracted_entities=entities,
            semantic_query=semantic_query
        )

    def add_to_conversation_memory(self, user_input: str, ai_response: str):
        """
        Add a user input and AI response to conversation memory.
        
        :param user_input: User's input/query
        :param ai_response: AI's response
        """
        self.conversation_memory.save_context(
            {"human_input": user_input},
            {"ai_response": ai_response}
        )
        
        # Update user context with current query
        self.user_context['previous_searches'].append({
            'query': user_input,
            'timestamp': datetime.now().isoformat(),
            'response_summary': ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
        })
        
        # Keep only last 20 searches to avoid memory bloat
        if len(self.user_context['previous_searches']) > 20:
            self.user_context['previous_searches'] = self.user_context['previous_searches'][-20:]

    def get_conversation_context(self) -> ConversationContext:
        """
        Get the current conversation context.
        
        :return: ConversationContext with current state
        """
        # Extract recent queries
        recent_queries = [search['query'] for search in self.user_context['previous_searches'][-5:]]
        
        # Get conversation history
        memory_variables = self.conversation_memory.load_memory_variables({})
        conversation_summary = self._summarize_conversation(memory_variables.get('chat_history', []))
        
        return ConversationContext(
            user_preferences=self.user_context['preferences'],
            previous_recommendations=self.user_context.get('previous_recommendations', []),
            current_session_queries=recent_queries,
            follow_up_context=self._extract_follow_up_context(),
            conversation_summary=conversation_summary
        )

    def _summarize_conversation(self, chat_history: List) -> str:
        """
        Create a summary of the conversation history.
        
        :param chat_history: List of chat messages
        :return: Summary string
        """
        if not chat_history:
            return "No previous conversation."
        
        # Simple summarization - can be enhanced with LLM if available
        user_queries = []
        for message in chat_history:
            if hasattr(message, 'content'):
                if isinstance(message, HumanMessage):
                    user_queries.append(message.content)
        
        if not user_queries:
            return "No previous queries in conversation."
        
        if len(user_queries) == 1:
            return f"User previously asked: {user_queries[0]}"
        else:
            return f"User has made {len(user_queries)} queries, most recent: {user_queries[-1]}"

    def _extract_follow_up_context(self) -> Dict[str, Any]:
        """
        Extract context for follow-up questions and references.
        
        :return: Dictionary with follow-up context
        """
        context = {
            'last_recommendations': [],
            'referenced_sets': [],
            'ongoing_conversation_theme': None
        }
        
        # Get last recommendations from user context
        if self.user_context.get('previous_recommendations'):
            context['last_recommendations'] = self.user_context['previous_recommendations'][-3:]
        
        # Extract themes from recent searches
        recent_searches = self.user_context.get('previous_searches', [])
        if recent_searches:
            themes = []
            for search in recent_searches[-3:]:
                query = search['query'].lower()
                for theme_keywords in ['star wars', 'city', 'technic', 'creator', 'friends', 'ninjago']:
                    if theme_keywords in query:
                        themes.append(theme_keywords.title())
            
            if themes:
                context['ongoing_conversation_theme'] = max(set(themes), key=themes.count)
        
        return context

    def update_user_preferences(self, preferences: Dict[str, Any]):
        """
        Update user preferences based on interactions.
        
        :param preferences: Dictionary of user preferences
        """
        self.user_context['preferences'].update(preferences)
        
    def record_user_feedback(self, set_num: str, feedback: str, rating: Optional[int] = None):
        """
        Record user feedback on recommended sets.
        
        :param set_num: LEGO set number
        :param feedback: User feedback (liked/disliked/interested)
        :param rating: Optional numerical rating
        """
        feedback_entry = {
            'set_num': set_num,
            'feedback': feedback,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        if feedback in ['liked', 'interested', 'want']:
            self.user_context['liked_sets'].append(feedback_entry)
        elif feedback in ['disliked', 'not_interested', 'too_expensive']:
            self.user_context['disliked_sets'].append(feedback_entry)
        
        # Learn from feedback to update preferences
        self._learn_from_feedback(set_num, feedback)

    def _learn_from_feedback(self, set_num: str, feedback: str):
        """
        Learn from user feedback to improve future recommendations.
        
        :param set_num: LEGO set number
        :param feedback: User feedback
        """
        try:
            # Get set details from database
            query = """
            SELECT s.name, s.theme_id, t.name as theme_name, s.num_parts, s.year
            FROM sets s
            LEFT JOIN themes t ON s.theme_id = t.id
            WHERE s.set_num = ?
            """
            
            result = pd.read_sql_query(query, self.dbconn, params=[set_num])
            
            if not result.empty:
                set_info = result.iloc[0]
                
                # Update preferences based on feedback
                if feedback in ['liked', 'interested']:
                    # Increase preference for this theme
                    theme = set_info['theme_name']
                    if theme:
                        current_pref = self.user_context['preferences'].get('themes', {})
                        current_pref[theme] = current_pref.get(theme, 0) + 1
                        self.user_context['preferences']['themes'] = current_pref
                    
                    # Learn piece count preference
                    pieces = set_info['num_parts']
                    if pieces:
                        piece_prefs = self.user_context['preferences'].get('piece_ranges', [])
                        piece_prefs.append(pieces)
                        # Keep only last 10 preferences
                        if len(piece_prefs) > 10:
                            piece_prefs = piece_prefs[-10:]
                        self.user_context['preferences']['piece_ranges'] = piece_prefs
                        
                elif feedback in ['disliked', 'not_interested']:
                    # Decrease preference for this theme
                    theme = set_info['theme_name']
                    if theme:
                        current_pref = self.user_context['preferences'].get('themes', {})
                        current_pref[theme] = current_pref.get(theme, 0) - 1
                        self.user_context['preferences']['themes'] = current_pref
                        
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")

    def clear_conversation_memory(self):
        """Clear conversation memory and reset user context."""
        self.conversation_memory.clear()
        self.user_context = {
            'preferences': {},
            'previous_searches': [],
            'liked_sets': [],
            'disliked_sets': [],
            'conversation_session': datetime.now().isoformat()
        }

    def process_nl_query_with_context(self, query: str, user_context: Optional[Dict] = None) -> NLQueryResult:
        """
        Process natural language query with conversation context.
        
        :param query: Natural language query string
        :param user_context: Optional additional user context
        :return: NLQueryResult with conversation context applied
        """
        # Get conversation context
        conv_context = self.get_conversation_context()
        
        # Enhance query with conversation context
        enhanced_query = self._enhance_query_with_context(query, conv_context)
        
        # Detect intent with conversation context
        intent = self._detect_intent_with_context(enhanced_query, conv_context)
        
        # Extract filters and entities with context
        if hasattr(self, 'llm') and self.llm:
            filters = self._extract_filters_llm_with_context(enhanced_query, conv_context)
            entities = self._extract_entities_llm_with_context(enhanced_query, conv_context)
        else:
            filters = self._extract_filters_regex_with_context(enhanced_query, conv_context)
            entities = self._extract_entities_regex_with_context(enhanced_query, conv_context)
        
        # Create semantic query with context
        semantic_query = self._create_semantic_query_with_context(enhanced_query, filters, entities, conv_context)
        
        # Calculate confidence with context
        confidence = self._calculate_confidence_with_context(enhanced_query, intent, filters, entities, conv_context)
        
        return NLQueryResult(
            intent=intent,
            filters=filters,
            confidence=confidence,
            extracted_entities=entities,
            semantic_query=semantic_query
        )

    def _enhance_query_with_context(self, query: str, conv_context: ConversationContext) -> str:
        """
        Enhance the query with conversation context for better understanding.
        
        :param query: Original query
        :param conv_context: Conversation context
        :return: Enhanced query string
        """
        enhanced_parts = [query]
        
        # Add context from previous queries if query seems to be a follow-up
        follow_up_indicators = ['that', 'it', 'similar', 'like that', 'same', 'also', 'another']
        if any(indicator in query.lower() for indicator in follow_up_indicators):
            if conv_context.current_session_queries:
                last_query = conv_context.current_session_queries[-1]
                enhanced_parts.append(f"Previous context: {last_query}")
        
        # Add ongoing theme context
        if conv_context.follow_up_context.get('ongoing_conversation_theme'):
            theme = conv_context.follow_up_context['ongoing_conversation_theme']
            enhanced_parts.append(f"User has been interested in {theme} theme")
        
        # Add user preferences
        if conv_context.user_preferences.get('themes'):
            preferred_themes = conv_context.user_preferences['themes']
            top_themes = sorted(preferred_themes.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_themes:
                theme_names = [theme for theme, _ in top_themes]
                enhanced_parts.append(f"User prefers themes: {', '.join(theme_names)}")
        
        return " | ".join(enhanced_parts)

    def _detect_intent_with_context(self, query: str, conv_context: ConversationContext) -> str:
        """
        Detect intent with conversation context.
        
        :param query: Query string
        :param conv_context: Conversation context
        :return: Detected intent
        """
        # Use base intent detection first
        base_intent = self._detect_intent(query)
        
        # Enhance with context
        if conv_context.previous_recommendations:
            # If user had recent recommendations, follow-up queries might be for similar items
            if any(word in query.lower() for word in ['similar', 'like', 'alternative', 'another']):
                return 'recommend_similar'
        
        return base_intent

    def _extract_filters_llm_with_context(self, query: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Extract filters using LLM with conversation context."""
        if not hasattr(self, 'llm') or not self.llm:
            return self._extract_filters_regex_with_context(query, conv_context)
        
        # Create context-aware prompt
        context_info = []
        if conv_context.user_preferences.get('themes'):
            context_info.append(f"User prefers themes: {conv_context.user_preferences['themes']}")
        if conv_context.previous_recommendations:
            context_info.append("User has received previous recommendations")
        
        context_string = " | ".join(context_info) if context_info else "No previous context"
        
        # Use the existing LLM filter extraction but with context
        parser = PydanticOutputParser(pydantic_object=SearchFilters)
        
        prompt = PromptTemplate(
            template="""Extract search filters from the following LEGO set query with conversation context. \n
            Context: {context} \n
            Query: {query} \n
            {format_instructions} \n
            Only extract filters that are explicitly mentioned in the query or clearly implied by the context. \n""",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        try:
            output = (prompt | self.llm).invoke({"query": query, "context": context_string})
            if hasattr(output, 'content'):
                output_text = output.content
            else:
                output_text = str(output)
            filters = parser.parse(output_text)
            try:
                return filters.model_dump(exclude_none=True)
            except AttributeError:
                return filters.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"Context-aware LLM filter extraction failed: {e}")
            return self._extract_filters_regex_with_context(query, conv_context)

    def _extract_filters_regex_with_context(self, query: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Extract filters using regex with conversation context."""
        # Start with base regex extraction
        filters = self._extract_filters_regex(query)
        
        # Apply context-based filter enhancement
        if not filters.get('themes') and conv_context.user_preferences.get('themes'):
            # If no themes specified but user has preferences, don't auto-apply
            # Let user be explicit about theme changes
            pass
        
        # Apply learned piece count preferences if no specific range given
        if not filters.get('min_pieces') and not filters.get('max_pieces'):
            piece_prefs = conv_context.user_preferences.get('piece_ranges', [])
            if piece_prefs:
                avg_pieces = sum(piece_prefs) / len(piece_prefs)
                # Create a reasonable range around user's average preference
                filters['min_pieces'] = int(avg_pieces * 0.7)
                filters['max_pieces'] = int(avg_pieces * 1.3)
        
        return filters

    def _extract_entities_llm_with_context(self, query: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Extract entities using LLM with conversation context."""
        if not hasattr(self, 'llm') or not self.llm:
            return self._extract_entities_regex_with_context(query, conv_context)
        
        # Use base entity extraction with context information
        entities = self._extract_entities_llm(query)
        
        # Enhance with conversation context
        if not entities.get('recipient') and conv_context.user_preferences.get('recipient'):
            entities['recipient'] = conv_context.user_preferences['recipient']
        
        return entities

    def _extract_entities_regex_with_context(self, query: str, conv_context: ConversationContext) -> Dict[str, Any]:
        """Extract entities using regex with conversation context."""
        # Start with base regex extraction
        entities = self._extract_entities_regex(query)
        
        # Apply context-based enhancement
        if not entities.get('recipient') and conv_context.user_preferences.get('recipient'):
            entities['recipient'] = conv_context.user_preferences['recipient']
        
        return entities

    def _create_semantic_query_with_context(self, query: str, filters: Dict, entities: Dict, conv_context: ConversationContext) -> str:
        """Create semantic query with conversation context."""
        # Start with base semantic query
        semantic_query = self._create_semantic_query(query, filters, entities)
        
        # Add context from previous recommendations
        if conv_context.previous_recommendations:
            recent_themes = [rec.get('theme', '') for rec in conv_context.previous_recommendations[-3:]]
            recent_themes = [t for t in recent_themes if t]  # Remove empty themes
            if recent_themes:
                semantic_query += f" | Previously interested in {', '.join(set(recent_themes))} themes"
        
        # Add user preference context
        if conv_context.user_preferences.get('themes'):
            preferred_themes = conv_context.user_preferences['themes']
            top_themes = sorted(preferred_themes.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_themes:
                theme_names = [theme for theme, _ in top_themes]
                semantic_query += f" | User generally prefers {', '.join(theme_names)}"
        
        return semantic_query

    def _calculate_confidence_with_context(self, query: str, intent: str, filters: Dict, entities: Dict, conv_context: ConversationContext) -> float:
        """Calculate confidence with conversation context."""
        # Start with base confidence calculation
        base_confidence = self._calculate_confidence(query, intent, filters, entities)
        
        # Boost confidence if we have relevant context
        context_boost = 0.0
        
        # Boost if user has established preferences
        if conv_context.user_preferences.get('themes'):
            context_boost += 0.05
        
        # Boost if this seems like a follow-up query
        if conv_context.current_session_queries and len(conv_context.current_session_queries) > 1:
            context_boost += 0.03
        
        # Boost if we have previous recommendations to reference
        if conv_context.previous_recommendations:
            context_boost += 0.02
        
        return min(1.0, base_confidence + context_boost)

    def _detect_intent(self, query: str) -> str:
        """
        Detect the intent of the user's query based on predefined intents.
        Prioritize more specific intents over general ones.

        :param query: Natural language query string
        :return: Detected intent as a string
        """        
        query_lower = query.lower()
        
        # Check more specific intents first (order matters)
        intent_priority = [
            'gift_recommendation',
            'recommend_similar', 
            'collection_advice',
            'search'  # Most general, check last
        ]
        
        for intent in intent_priority:
            keywords = self.intents.get(intent, [])
            if any(keyword in query_lower for keyword in keywords):
                return intent
            
        return 'search'
    
    def _extract_filters_llm(self, query: str) -> Dict[str, Any]:
        """
        Extract search filters using LLM.

        :param query: Natural language query string
        :return: SearchFilters object with extracted filters
        """
        # Implement local filter extraction logic
        if not hasattr(self, 'llm') or not self.llm:
            return self._extract_filters_regex(query)
        
        # Creat output parser
        # Create output parser
        parser = PydanticOutputParser(pydantic_object=SearchFilters)

        # Create prompt
        prompt = PromptTemplate(
            template="""Extract search filters from the following LEGO set query. \n
            Query: {query} {format_instructions} \n
            Only extract filters that are explicitly mentioned in the query. \n""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Create chain using modern approach
        try:
            # Use new RunnableSequence approach (prompt | llm)
            output = (prompt | self.llm).invoke({"query": query})
            # Handle different output types
            if hasattr(output, 'content'):
                output_text = output.content
            else:
                output_text = str(output)
            filters = parser.parse(output_text)
            # Handle different Pydantic versions
            try:
                return filters.model_dump(exclude_none=True)  # Pydantic v2
            except AttributeError:
                return filters.dict(exclude_none=True)  # Pydantic v1
        except Exception as e:
            logger.error(f"LLM filter extraction failed: {e}")
            return self._extract_filters_regex(query)
        
    def _load_available_themes(self) -> List[str]:
        """Load all available themes from database (cached)"""
        # Cache themes to avoid repeated database queries
        if not hasattr(self, '_cached_themes'):
            query = "SELECT DISTINCT LOWER(name) as theme_name FROM themes WHERE name IS NOT NULL"
            df = pd.read_sql_query(query, self.dbconn)
            self._cached_themes = df['theme_name'].tolist()
        return self._cached_themes
            
    def _extract_filters_regex(self, query: str) -> Dict[str, Any]:
        """
        Extract search filters using regex patterns.

        :param query: Natural language query string
        :return: SearchFilters object with extracted filters
        """
        import re
        filters = {}
        piece_patterns = [
            r'between\s+(\d+)\s+and\s+(\d+)\s*pieces?',
            r'(\d+)\s*(?:to|-)\s*(\d+)\s*pieces?',
            r'(?:under|less than|<)\s*(\d+)\s*pieces?',
            r'(?:over|more than|>)\s*(\d+)\s*pieces?',
            r'(\d+)\s*pieces?'
        ]

        for pattern in piece_patterns:
            match = re.search(pattern, query.lower())
            if match:
                if len(match.groups()) == 2 and match.group(2):
                    filters['min_pieces'] = int(match.group(1))
                    filters['max_pieces'] = int(match.group(2))
                elif 'under' in pattern or 'less' in pattern:
                    filters['max_pieces'] = int(match.group(1))
                elif 'over' in pattern or 'more' in pattern:
                    filters['min_pieces'] = int(match.group(1))
                else:
                    # Assume approximate range
                    pieces = int(match.group(1))
                    filters['min_pieces'] = int(pieces * 0.8)
                    filters['max_pieces'] = int(pieces * 1.2)
                break
        
        # Extract age
        age_match = re.search(r'(\d+)[\s-]*(?:year|yr)[\s-]*old', query.lower())
        if age_match:
            age = int(age_match.group(1))
            filters['min_age'] = max(age - 2, 4)
            filters['max_age'] = age + 2
        
        # Extract themes
        known_themes = self._load_available_themes()
        
        found_themes = []
        for theme in known_themes:
            if theme in query.lower():
                found_themes.append(theme.title())
        
        if found_themes:
            filters['themes'] = found_themes
        
        # Extract complexity
        if any(word in query.lower() for word in ['simple', 'easy', 'beginner']):
            filters['complexity'] = 'simple'
        elif any(word in query.lower() for word in ['complex', 'challenging', 'advanced', 'expert']):
            filters['complexity'] = 'complex'
        elif any(word in query.lower() for word in ['moderate', 'intermediate']):
            filters['complexity'] = 'moderate'
        
        # Extract budget
        budget_match = re.search(r'\$(\d+)(?:\s*(?:to|-)\s*\$?(\d+))?', query)
        if budget_match:
            if budget_match.group(2):
                filters['budget_min'] = float(budget_match.group(1))
                filters['budget_max'] = float(budget_match.group(2))
            else:
                # Single price mentioned, create range
                price = float(budget_match.group(1))
                filters['budget_min'] = price * 0.7
                filters['budget_max'] = price * 1.3
        
        return filters
        
    def _extract_entities_llm(self, query: str) -> Dict[str, Any]:
        """Extract entities using LLM"""
        if not hasattr(self, 'llm') or not self.llm:
            return self._extract_entities_regex(query)
        
        # Create a comprehensive entity extraction prompt
        entity_prompt = PromptTemplate(
            template="""Extract relevant entities from the following LEGO set query. 
            Focus on identifying specific entities that would help with LEGO set recommendations.

            Query: {query}

            Extract the following entities if mentioned in the query:
            1. Recipient: Who is this for? (e.g., "son", "daughter", "nephew", "friend", "myself")
            2. Age: Specific age mentioned for the recipient
            3. Occasion: Special event or holiday (e.g., "birthday", "christmas", "graduation")
            4. Building preference: Building style preferences (e.g., "detailed", "quick_build", "challenging")
            5. Experience level: Builder experience (e.g., "beginner", "expert", "intermediate")
            6. Interest category: General interests (e.g., "vehicles", "buildings", "fantasy", "space")
            7. Time constraint: Available building time (e.g., "weekend project", "quick build")
            8. Special features: Desired features (e.g., "motorized", "lights", "minifigures")

            Return the results in JSON format with only the entities that are explicitly mentioned or clearly implied.
            If an entity is not present, do not include it in the response.

            Example output:
            {{
                "recipient": "son",
                "age": 8,
                "occasion": "birthday",
                "building_preference": "detailed",
                "experience_level": "beginner"
            }}

            JSON Response:""",
            input_variables=["query"]
        )
        
        try:
            # Use the modern LangChain approach
            output = (entity_prompt | self.llm).invoke({"query": query})
            
            # Handle different output types
            if hasattr(output, 'content'):
                output_text = output.content
            else:
                output_text = str(output)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from the response (in case there's extra text)
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                entities = json.loads(json_str)
                
                # Validate and clean the entities
                cleaned_entities = {}
                for key, value in entities.items():
                    if value is not None and value != "":
                        # Convert age to integer if it's a string
                        if key == "age" and isinstance(value, str):
                            try:
                                cleaned_entities[key] = int(value)
                            except ValueError:
                                continue
                        # Normalize time_constraint values
                        elif key == "time_constraint":
                            if isinstance(value, str):
                                # Normalize "weekend project" to "weekend_project"
                                normalized_value = value.lower().replace(" ", "_")
                                cleaned_entities[key] = normalized_value
                            else:
                                cleaned_entities[key] = value
                        # Ensure special_features is always a list
                        elif key == "special_features":
                            if isinstance(value, str):
                                cleaned_entities[key] = [value]
                            elif isinstance(value, list):
                                cleaned_entities[key] = value
                            else:
                                cleaned_entities[key] = [str(value)]
                        else:
                            cleaned_entities[key] = value
                
                logger.debug(f"LLM extracted entities: {cleaned_entities}")
                return cleaned_entities
            else:
                logger.warning("No valid JSON found in LLM response, falling back to regex")
                return self._extract_entities_regex(query)
                
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            logger.debug(f"LLM output was: {output_text if 'output_text' in locals() else 'No output'}")
            return self._extract_entities_regex(query)
    
    def _extract_entities_regex(self, query: str) -> Dict[str, Any]:
        """Extract named entities using patterns"""
        import re
        entities = {}
        
        # Extract recipient info
        recipient_patterns = [
            r'for (?:my |a )?(\w+)',
            r'(?:my |a )(\d+)[\s-]*year[\s-]*old',
            r'(\w+)\'s birthday',
        ]
        
        for pattern in recipient_patterns:
            recipient_match = re.search(pattern, query.lower())
            if recipient_match:
                entities['recipient'] = recipient_match.group(1)
                break
        
        # Extract age more comprehensively
        age_patterns = [
            r'(\d+)[\s-]*(?:year|yr)[\s-]*old',
            r'age\s*(\d+)',
            r'(\d+)\s*years?\s*old',
            r'for\s*a\s*(\d+)',
        ]
        
        for pattern in age_patterns:
            age_match = re.search(pattern, query.lower())
            if age_match:
                entities['age'] = int(age_match.group(1))
                break
        
        # Extract occasion
        occasions = {
            'birthday': ['birthday', 'b-day', 'bday'],
            'christmas': ['christmas', 'xmas', 'holiday'],
            'graduation': ['graduation', 'grad'],
            'anniversary': ['anniversary'],
            'gift': ['gift', 'present']
        }
        
        for occasion, keywords in occasions.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['occasion'] = occasion
                break
        
        # Extract building preferences
        building_prefs = {
            'detailed': ['detail', 'detailed', 'intricate', 'complex'],
            'quick_build': ['quick', 'fast', 'simple', 'easy'],
            'challenging': ['challenging', 'difficult', 'advanced', 'expert']
        }
        
        for pref, keywords in building_prefs.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['building_preference'] = pref
                break
        
        # Extract experience level
        experience_levels = {
            'beginner': ['beginner', 'new', 'first time', 'starting'],
            'intermediate': ['intermediate', 'moderate', 'some experience'],
            'expert': ['expert', 'advanced', 'experienced', 'pro']
        }
        
        for level, keywords in experience_levels.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['experience_level'] = level
                break
        
        # Extract interest categories
        interests = {
            'vehicles': ['car', 'truck', 'plane', 'ship', 'vehicle', 'transport'],
            'buildings': ['house', 'building', 'castle', 'architecture'],
            'space': ['space', 'rocket', 'spaceship', 'astronaut'],
            'fantasy': ['dragon', 'wizard', 'magic', 'fantasy'],
            'robots': ['robot', 'mech', 'android'],
            'animals': ['animal', 'zoo', 'pet', 'wildlife']
        }
        
        for category, keywords in interests.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['interest_category'] = category
                break
        
        # Extract special features
        special_features = []
        feature_keywords = {
            'motorized': ['motor', 'motorized', 'moves'],
            'lights': ['light', 'led', 'glows'],
            'minifigures': ['minifig', 'figure', 'character'],
            'remote_control': ['remote', 'rc', 'control'],
            'sound': ['sound', 'noise', 'music']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                special_features.append(feature)
        
        if special_features:
            entities['special_features'] = special_features
        
        # Extract time constraints
        time_patterns = {
            'weekend_project': ['weekend', 'saturday', 'sunday'],
            'quick_build': ['hour', 'quick', 'fast'],
            'long_project': ['weeks', 'months', 'long project']
        }
        
        for time_type, keywords in time_patterns.items():
            if any(keyword in query.lower() for keyword in keywords):
                entities['time_constraint'] = time_type
                break
        
        return entities
    
    def _create_semantic_query(self, og_query: str, filters: Dict, entities: Dict) -> str:
        """
        Create a semantic query for embedding search based on original query, filters, and entities.

        :param og_query: Original natural language query
        :param filters: Extracted search filters
        :param entities: Extracted named entities
        :return: Semantic query string
        """
        query_parts = [og_query]
        
        # Add filter context
        if filters.get('themes'):
            query_parts.append(f"from {', '.join(filters['themes'])} themes")
        
        if filters.get('complexity'):
            query_parts.append(f"{filters['complexity']} complexity building experience")
        
        # Add entity context for better semantic matching
        if entities.get('recipient'):
            query_parts.append(f"suitable for {entities['recipient']}")
        
        if entities.get('age'):
            age = entities['age']
            if age <= 6:
                query_parts.append("simple builds for young children")
            elif age <= 12:
                query_parts.append("engaging builds for kids")
            elif age <= 16:
                query_parts.append("challenging builds for teenagers")
            else:
                query_parts.append("sophisticated builds for adults")
        
        if entities.get('occasion'):
            occasion = entities['occasion']
            if occasion == 'birthday':
                query_parts.append("special birthday present")
            elif occasion == 'christmas':
                query_parts.append("holiday gift")
            else:
                query_parts.append(f"{occasion} gift")
        
        if entities.get('building_preference'):
            pref = entities['building_preference']
            if pref == 'detailed':
                query_parts.append("intricate detailed construction")
            elif pref == 'quick_build':
                query_parts.append("quick easy assembly")
            elif pref == 'challenging':
                query_parts.append("complex challenging build")
        
        if entities.get('experience_level'):
            level = entities['experience_level']
            query_parts.append(f"appropriate for {level} builders")
        
        if entities.get('interest_category'):
            category = entities['interest_category']
            query_parts.append(f"{category} themed sets")
        
        if entities.get('special_features'):
            features = entities['special_features']
            # Ensure features is a list
            if isinstance(features, str):
                features = [features]
            elif not isinstance(features, list):
                features = [str(features)]
            query_parts.append(f"featuring {', '.join(features)}")
        
        if entities.get('time_constraint'):
            time_constraint = entities['time_constraint']
            # Normalize time constraint format
            normalized_constraint = str(time_constraint).lower().replace(" ", "_")
            if normalized_constraint == 'weekend_project':
                query_parts.append("suitable for weekend building")
            elif normalized_constraint == 'quick_build':
                query_parts.append("quick one-hour build")
            elif normalized_constraint == 'long_project':
                query_parts.append("extended building project")
        
        return " ".join(query_parts)
    
    def semantic_search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
         """
         Perform semantic search for LEGO sets
         :param query: Natural language query string
         :param top_k: Number of top results to return
         :param filters: Optional search filters to apply
         :return: List of dictionaries with search results
         """
         if not self.vectorstore:
             self.prep_vectorDB()

         # Process the NL query
         nl_result = self.process_nl_query(query, None)

         # Use the semantic query for search - using modern invoke method
         results = self.retriever.invoke(nl_result.semantic_query)

         # Apply filters if provided
         filtered_results = self._apply_filters(results, nl_result.filters)
         
         # Convert documents to result format
         search_results = []
         for doc in filtered_results[:top_k]:
             search_results.append({
                 'set_num': doc.metadata['set_num'],
                 'name': doc.metadata['name'],
                 'year': doc.metadata['year'],
                 'num_parts': doc.metadata['num_parts'],
                 'theme': doc.metadata['theme'],
                 'description': doc.page_content,  # Add description from document content
                 'score': doc.metadata.get('score', 0.0)
             })
         
         return search_results

    def semantic_search_with_context(self, query: str, top_k: int = 10, filters: Optional[Dict] = None, 
                                   record_in_memory: bool = True) -> List[Dict]:
        """
        Perform semantic search with conversation context awareness.
        
        :param query: Natural language query string
        :param top_k: Number of top results to return
        :param filters: Optional search filters to apply
        :param record_in_memory: Whether to record this interaction in conversation memory
        :return: List of dictionaries with search results
        """
        if not self.vectorstore:
            self.prep_vectorDB()

        # Process the NL query with conversation context
        nl_result = self.process_nl_query_with_context(query, None)

        # Use the semantic query for search
        results = self.retriever.invoke(nl_result.semantic_query)

        # Apply filters
        filtered_results = self._apply_filters(results, nl_result.filters)
        
        # Apply user preference-based ranking
        ranked_results = self._apply_preference_ranking(filtered_results)
        
        # Convert documents to result format
        search_results = []
        for doc in ranked_results[:top_k]:
            result = {
                'set_num': doc.metadata['set_num'],
                'name': doc.metadata['name'],
                'year': doc.metadata['year'],
                'num_parts': doc.metadata['num_parts'],
                'theme': doc.metadata['theme'],
                'description': doc.page_content,
                'score': doc.metadata.get('score', 0.0),
                'confidence': nl_result.confidence,
                'intent': nl_result.intent
            }
            search_results.append(result)
        
        # Record this interaction in conversation memory if requested
        if record_in_memory:
            response_summary = f"Found {len(search_results)} LEGO sets"
            if search_results:
                top_themes = list(set(r['theme'] for r in search_results[:3]))
                response_summary += f" mainly from {', '.join(top_themes)} themes"
            
            self.add_to_conversation_memory(query, response_summary)
            
            # Update user context with recommendations
            self.user_context.setdefault('previous_recommendations', []).extend(search_results)
            # Keep only last 50 recommendations
            if len(self.user_context['previous_recommendations']) > 50:
                self.user_context['previous_recommendations'] = self.user_context['previous_recommendations'][-50:]
        
        return search_results

    def _apply_preference_ranking(self, results: List[Document]) -> List[Document]:
        """
        Apply user preference-based ranking to search results.
        
        :param results: List of Document objects
        :return: Ranked list of Document objects
        """
        if not self.user_context.get('preferences'):
            return results
        
        # Get user theme preferences
        theme_prefs = self.user_context['preferences'].get('themes', {})
        
        # Score results based on preferences
        scored_results = []
        for doc in results:
            preference_score = 0.0
            
            # Theme preference scoring
            doc_theme = doc.metadata.get('theme', '')
            if doc_theme in theme_prefs:
                preference_score += theme_prefs[doc_theme] * 0.1
            
            # Piece count preference scoring
            piece_prefs = self.user_context['preferences'].get('piece_ranges', [])
            if piece_prefs:
                avg_preferred_pieces = sum(piece_prefs) / len(piece_prefs)
                doc_pieces = doc.metadata.get('num_parts', 0)
                # Score based on how close to user's preferred piece count
                if doc_pieces > 0:
                    piece_diff = abs(doc_pieces - avg_preferred_pieces) / avg_preferred_pieces
                    preference_score += max(0, 1 - piece_diff) * 0.05
            
            # Add preference score to metadata
            doc.metadata['preference_score'] = preference_score
            scored_results.append(doc)
        
        # Sort by preference score (descending)
        scored_results.sort(key=lambda x: x.metadata.get('preference_score', 0), reverse=True)
        return scored_results

    def _apply_filters(self, results: List[Document], filters: Dict) -> List[Document]:
        """
        Apply extracted filters to search results
        :param results: List of Document objects from the vector store
        :param filters: Dictionary of search filters
        :return: Filtered list of Document objects
        """
        filtered = []
        
        for doc in results:
            # Check piece count
            if filters.get('min_pieces') and doc.metadata['num_parts'] < filters['min_pieces']:
                continue
            if filters.get('max_pieces') and doc.metadata['num_parts'] > filters['max_pieces']:
                continue
            
            # Check theme
            if filters.get('themes'):
                theme_match = any(theme.lower() in doc.metadata['theme'].lower() 
                                for theme in filters['themes'])
                if not theme_match:
                    continue
            
            # Check complexity
            if filters.get('complexity') and doc.metadata.get('complexity') != filters['complexity']:
                continue
            
            # Check year (approximate age)
            if filters.get('min_age'):
                # Newer sets for younger builders
                if doc.metadata['year'] < 2010:
                    continue
            
            filtered.append(doc)
        
        return filtered
    
    def create_recommendation_prompts(self):
        """
        Create prompt templates for different recommendation scenarios.
        """
        self.prompts = {
            'gift_recommendation': PromptTemplate(
                template=(
                    "You are a LEGO expert helping someone find the perfect gift.\n\n"
                    "User query: {query}\n"
                    "Recipient: {recipient}\n"
                    "Occasion: {occasion}\n"
                    "Budget: ${budget_min} - ${budget_max}\n\n"
                    "Available sets:\n"
                    "{available_sets}\n\n"
                    "Recommend the top 3 sets and explain why each would make a great gift."
                ),
                input_variables=[
                    "query", "recipient", "occasion", "budget_min", "budget_max", "available_sets"
                ]
            ),
            'collection_advice': PromptTemplate(
                template=(
                    "User query: {query}\n"
                    "Current collection themes: {current_themes}\n"
                    "Collection size: {collection_size} sets\n\n"
                    "Available sets:\n"
                    "{available_sets}\n\n"
                    "Provide advice on which sets would best complement their collection and why."
                ),
                input_variables=[
                    "query", "current_themes", "collection_size", "available_sets"
                ]
            ),
            'building_challenge': PromptTemplate(
                template=(
                    "You are helping a LEGO enthusiast find their next building challenge.\n\n"
                    "User query: {query}\n"
                    "Previous builds: {previous_builds}\n"
                    "Skill level: {skill_level}\n\n"
                    "Available sets:\n"
                    "{available_sets}\n\n"
                    "Recommend sets that will provide an appropriate challenge and explain the building experience."
                ),
                input_variables=[
                    "query", "previous_builds", "skill_level", "available_sets"
                ]
            )
        }

    def save_embeddings(self, path: str):
        """Save vector store to disk (PGVector persists automatically)"""
        if hasattr(self.vectorstore, 'save_local'):
            # For compatibility with other vector stores
            self.vectorstore.save_local(path)
        else:
            # PGVector stores data in PostgreSQL automatically
            logger.info("PGVector data is automatically persisted in PostgreSQL database")
    
    def load_embeddings(self, path: str):
        """Load or connect to existing vector store"""
        try:
            if self.use_openai:
                # For OpenAI, try to connect to existing PGVector collection
                try:
                    self.vectorstore = PGVector(
                        embeddings=self.embeddings,
                        collection_name="lego_sets",
                        connection=self.db_connection_string,
                        use_jsonb=True,
                    )
                    logger.info("Connected to existing PGVector collection")
                except Exception as e:
                    logger.warning(f"Could not connect to PGVector: {e}, falling back to Chroma")
                    self.vectorstore = Chroma(
                        persist_directory=path,
                        embedding_function=self.embeddings
                    )
            else:
                # For local embeddings, try to connect to existing PGVector collection
                try:
                    self.vectorstore = PGVector(
                        embeddings=self.embeddings,
                        collection_name="lego_sets", 
                        connection=self.db_connection_string,
                        use_jsonb=True,
                    )
                    logger.info("Connected to existing PGVector collection")
                except Exception as e:
                    logger.warning(f"Could not connect to PGVector: {e}")
                    return False
            
            if self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 20}
                )
                logger.info("Vector store connected successfully")
                return True
            else:
                logger.warning("Failed to load embeddings")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def generate_recommendation_explanation(self, query: str, top_results: List[Dict]) -> str:
        """
        Generate a human-readable explanation for the recommendation results
        
        :param query: Original query string
        :param top_results: List of top recommendation results
        :return: Explanation string
        """
        if not top_results:
            return "No matching sets were found for your query."
        
        # Extract key themes and features from results
        themes = list(set(result.get('theme', '') for result in top_results))
        themes = [t for t in themes if t]  # Remove empty themes
        
        avg_pieces = sum(result.get('num_parts', 0) for result in top_results) / len(top_results)
        
        explanation_parts = [
            f"Based on your search for '{query}', I found {len(top_results)} relevant LEGO sets."
        ]
        
        if themes:
            if len(themes) == 1:
                explanation_parts.append(f"All recommendations are from the {themes[0]} theme.")
            else:
                explanation_parts.append(f"These sets span multiple themes: {', '.join(themes[:3])}.")
        
        if avg_pieces > 0:
            if avg_pieces < 200:
                complexity = "smaller, quick-build"
            elif avg_pieces > 1000:
                complexity = "large, detailed"
            else:
                complexity = "medium-sized"
            
            explanation_parts.append(f"The average set size is {int(avg_pieces)} pieces, making these {complexity} sets.")
        
        # Add specific mentions for standout features
        if any('detail' in query.lower() for query in [query]):
            explanation_parts.append("These sets were selected for their detailed construction and visual appeal.")
        
        return " ".join(explanation_parts)

    def get_conversational_recommendations(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Get recommendations with full conversation context and explanations.
        
        :param query: Natural language query string
        :param top_k: Number of top results to return
        :return: Dictionary with recommendations and conversation context
        """
        # Get conversation context
        conv_context = self.get_conversation_context()
        
        # Perform context-aware search
        results = self.semantic_search_with_context(query, top_k, record_in_memory=True)
        
        # Generate explanation
        explanation = self.generate_recommendation_explanation(query, results)
        
        # Add conversation-specific explanation elements
        if conv_context.current_session_queries:
            if len(conv_context.current_session_queries) > 1:
                explanation += f" This builds on your previous {len(conv_context.current_session_queries)} queries in this session."
        
        if conv_context.user_preferences.get('themes'):
            preferred_themes = conv_context.user_preferences['themes']
            top_theme = max(preferred_themes.items(), key=lambda x: x[1])[0]
            explanation += f" Based on your history, you tend to prefer {top_theme} sets."
        
        return {
            'query': query,
            'results': results,
            'explanation': explanation,
            'conversation_context': {
                'session_queries': conv_context.current_session_queries,
                'user_preferences': conv_context.user_preferences,
                'conversation_summary': conv_context.conversation_summary
            },
            'recommendation_metadata': {
                'total_results': len(results),
                'confidence': results[0]['confidence'] if results else 0.0,
                'intent': results[0]['intent'] if results else 'search',
                'timestamp': datetime.now().isoformat()
            }
        }

    def handle_follow_up_query(self, query: str) -> Dict[str, Any]:
        """
        Handle follow-up queries that reference previous recommendations.
        
        :param query: Follow-up query string
        :return: Dictionary with follow-up response
        """
        conv_context = self.get_conversation_context()
        
        # Check if this is a follow-up query
        follow_up_indicators = ['that', 'it', 'those', 'them', 'similar', 'like that', 'same', 'also', 'another']
        is_follow_up = any(indicator in query.lower() for indicator in follow_up_indicators)
        
        if is_follow_up and conv_context.previous_recommendations:
            # Reference previous recommendations
            last_recs = conv_context.previous_recommendations[-5:]  # Last 5 recommendations
            
            # Create a follow-up specific response
            if 'similar' in query.lower() or 'like that' in query.lower():
                # Find similar sets to the previous recommendations
                similar_queries = []
                for rec in last_recs:
                    similar_queries.append(f"sets similar to {rec['name']} from {rec['theme']}")
                
                enhanced_query = f"{query} | {' | '.join(similar_queries)}"
                results = self.semantic_search_with_context(enhanced_query, record_in_memory=True)
                
                explanation = f"Based on your interest in {', '.join([r['name'] for r in last_recs[:3]])}, here are similar recommendations:"
                
            else:
                # General follow-up
                results = self.semantic_search_with_context(query, record_in_memory=True)
                explanation = f"Following up on your previous searches, here are relevant recommendations:"
            
            return {
                'query': query,
                'results': results,
                'explanation': explanation,
                'is_follow_up': True,
                'referenced_recommendations': last_recs
            }
        else:
            # Not a follow-up, handle as normal query
            return self.get_conversational_recommendations(query)