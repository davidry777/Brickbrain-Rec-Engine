import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticToolsParser, OutputFixingParser
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
    def __init__(self, dbcon, openai_flag: bool = False):
        self.dbconn = dbcon
        self.openai_flag = openai_flag

        # Initialize embeddings
        if self.openai_flag:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            self.llm = None # No LLM for local embeddings

        self.vectorstore = None
        self.retriever = None
        self.docs = []
        self.intents = {
            'search': ['find', 'search', 'looking for', 'show me', 'want', 'need'],
            'recommend_similar': ['similar to', 'like', 'comparable', 'alternative to'],
            'gift_recommendation': ['gift', 'present', 'birthday', 'christmas', 'for my', 'for a'],
            'collection_advice': ['should i buy', 'worth it', 'good investment', 'collection']
        }

    def prep_vectorDB(self):
        """
        Prepare vector database with LEGO set descriptions and metadata
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
        """

        df = pd.read_sql_query(query, self.dbconn)

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

        # Create vector store
        if self.openai_flag:
            self.vectorstore = Chroma.from_documents(
                documents=self.docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            self.vectorstore = FAISS.from_documents(
                documents=self.docs,
                embedding=self.embeddings
            )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
        logger.info("Vector database prepared with %d documents", len(self.docs))

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

