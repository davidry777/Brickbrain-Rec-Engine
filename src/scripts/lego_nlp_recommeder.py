import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# LangChain imports (updated to avoid deprecation warnings)
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.chains.llm import LLMChain

from langchain_openai.chat_models.base import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

# For local embeddings (no API required)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class NLQueryResult:
    """Result from natural language query processing"""
    intent: str
    filters: Dict[str, Any]
    confidence: float
    extracted_entities: Dict[str, Any]
    semantic_query: str

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

        # Initialize embeddings
        if self.use_openai:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        else:
            # Use free Hugging Face model for embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                },
                encode_kwargs={"normalize_embeddings": True}
            )
            self.llm = None  # No LLM for local embeddings - using pattern matching instead

        self.vectorstore = None
        self.retriever = None
        self.docs = []
        self.intents = {
            'search': ['find', 'search', 'looking for', 'show me', 'want', 'need'],
            'recommend_similar': ['similar to', 'like', 'comparable', 'alternative to'],
            'gift_recommendation': ['gift', 'present', 'birthday', 'christmas', 'for my', 'for a'],
            'collection_advice': ['should i buy', 'worth it', 'good investment', 'collection']
        }

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
        """Create vector store locally (for development/small datasets)"""
        # Create vector store
        if self.use_openai:
            self.vectorstore = Chroma.from_documents(
                documents=self.docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            # Process in smaller batches for FAISS to avoid memory issues
            batch_size = 50  # Smaller batches for local processing
            if len(self.docs) > batch_size:
                logger.info(f"Processing {len(self.docs)} documents in batches of {batch_size}")
                
                # Create initial vector store with first batch
                first_batch = self.docs[:batch_size]
                self.vectorstore = FAISS.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings
                )
                
                # Add remaining documents in batches
                for i in range(batch_size, len(self.docs), batch_size):
                    batch = self.docs[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.docs) + batch_size - 1)//batch_size}")
                    batch_vectorstore = FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                    self.vectorstore.merge_from(batch_vectorstore)
            else:
                self.vectorstore = FAISS.from_documents(
                    documents=self.docs,
                    embedding=self.embeddings
                )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        logger.info("Vector database prepared with %d documents", len(self.docs))
    
    def load_cloud_embeddings(self, embeddings_path: str, metadata_path: str):
        """
        Load pre-computed embeddings from cloud processing
        
        :param embeddings_path: Path to numpy array of embeddings
        :param metadata_path: Path to JSON file with metadata
        """
        import json
        import numpy as np
        
        # Load embeddings and metadata
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        # Create documents
        documents = []
        for i, meta in enumerate(metadata_list):
            doc = Document(
                page_content=meta['content'],
                metadata=meta['metadata']
            )
            documents.append(doc)
        
        # Create FAISS index from pre-computed embeddings
        dimension = embeddings.shape[1]
        import faiss
        
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # Create FAISS vectorstore with pre-computed index
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=faiss.InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
            index_to_docstore_id={i: str(i) for i in range(len(documents))}
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        self.docs = documents
        
        logger.info(f"Loaded cloud-processed vector database with {len(documents)} documents")
    
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
            entity_confidence = min(0.2, entity_types * 0.07)
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
        if self.llm:
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

    def _detect_intent(self, query: str) -> str:
        """
        Detect the intent of the user's query based on predefined intents.

        :param query: Natural language query string
        :return: Detected intent as a string
        """        
        for intent, keywords in self.intents.items():
            if any(keyword in query.lower() for keyword in keywords):
                return intent
            
        return 'search'
    
    def _extract_filters_llm(self, query: str) -> Dict[str, Any]:
        """
        Extract search filters using LLM.

        :param query: Natural language query string
        :return: SearchFilters object with extracted filters
        """
        # Implement local filter extraction logic
        if not self.llm:
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

        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # Run extraction
            output = chain.run(query=query)
            filters = parser.parse(output)
            return filters.dict(exclude_none=True)
        except Exception as e:
            logger.error(f"LLM filter extraction failed: {e}")
            return self._extract_filters_regex(query)
        
    def _load_available_themes(self) -> List[str]:
        """Load all available themes from database"""
        query = "SELECT DISTINCT LOWER(name) as theme_name FROM themes WHERE name IS NOT NULL"
        df = pd.read_sql_query(query, self.dbconn)
        return df['theme_name'].tolist()
            
    def _extract_filters_regex(self, query: str) -> Dict[str, Any]:
        """
        Extract search filters using regex patterns.

        :param query: Natural language query string
        :return: SearchFilters object with extracted filters
        """
        import re
        filters = {}
        piece_patterns = [
            r'(\d+)\s*(?:to|-)\s*(\d+)\s*pieces?',
            r'(?:under|less than|<)\s*(\d+)\s*pieces?',
            r'(?:over|more than|>)\s*(\d+)\s*pieces?',
            r'(\d+)\s*pieces?'
        ]

        for pattern in piece_patterns:
            match = re.search(pattern, query.lower())
            if match:
                if len(match.groups()) == 2:
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
        if not self.llm:
            return self._extract_entities_regex(query)
        
        # Would implement LLM-based entity extraction
        # For now, falling back to regex
        return self._extract_entities_regex(query)
    
    def _extract_entities_regex(self, query: str) -> Dict[str, Any]:
        """Extract named entities using patterns"""
        import re
        entities = {}
        
        # Extract recipient info
        recipient_match = re.search(r'for (?:my |a )?(\w+)', query.lower())
        if recipient_match:
            entities['recipient'] = recipient_match.group(1)
        
        # Extract occasion
        occasions = ['birthday', 'christmas', 'holiday', 'anniversary']
        for occasion in occasions:
            if occasion in query.lower():
                entities['occasion'] = occasion
                break
        
        # Extract building preferences
        if 'detail' in query.lower():
            entities['preference'] = 'detailed'
        elif 'quick' in query.lower() or 'fast' in query.lower():
            entities['preference'] = 'quick_build'
        
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
        
        if entities.get('recipient'):
            query_parts.append(f"suitable for {entities['recipient']}")
        
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

         # Use the semantic query for search
         results = self.retriever.get_relevant_documents(nl_result.semantic_query)

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
                 'score': doc.metadata.get('score', 0.0)
             })
         
         return search_results

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
        """Save vector store to disk"""
        if isinstance(self.vectorstore, FAISS):
            self.vectorstore.save_local(path)
        elif isinstance(self.vectorstore, Chroma):
            # Chroma persists automatically
            pass
    
    def load_embeddings(self, path: str):
        """Load vector store from disk"""
        if self.use_openai:
            self.vectorstore = Chroma(
                persist_directory=path,
                embedding_function=self.embeddings
            )
        elif os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings)
        
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}
            )