import logging
import json
import sqlite3
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, deque, Counter

# HuggingFace imports for summarization
from transformers import pipeline, AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    timestamp: str
    user_message: str
    assistant_response: str
    intent: str
    entities: Dict[str, Any]
    filters: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    user_feedback: Optional[str] = None
    confidence: float = 0.0

@dataclass
class UserProfile:
    """User profile with learned preferences"""
    user_id: str
    preferred_themes: Dict[str, float]  # theme -> preference score
    preferred_complexity: str  # beginner, intermediate, advanced
    typical_piece_range: Tuple[int, int]  # (min_pieces, max_pieces)
    budget_range: Tuple[float, float]  # (min_price, max_price)
    favorite_recipients: List[str]  # who they buy for
    building_style: str  # detailed, quick, display, play
    interaction_count: int = 0
    last_active: Optional[str] = None

class ConversationMemoryDB:
    """SQLite-based conversation memory with HuggingFace integration"""
    
    def __init__(self, db_path: str = "conversation_memory.db"):
        """
        Initialize conversation memory database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Initialize summarization pipeline
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU for now to save GPU memory
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {e}")
            self.summarizer = None
        
        self._create_tables()
        
    def _create_tables(self):
        """Create database tables for conversation memory"""
        cursor = self.connection.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                intent TEXT,
                entities TEXT,  -- JSON string
                filters TEXT,   -- JSON string
                recommendations TEXT,  -- JSON string
                user_feedback TEXT,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferred_themes TEXT,  -- JSON string
                preferred_complexity TEXT,
                min_pieces INTEGER,
                max_pieces INTEGER,
                min_price REAL,
                max_price REAL,
                favorite_recipients TEXT,  -- JSON string
                building_style TEXT,
                interaction_count INTEGER DEFAULT 0,
                last_active TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Conversation summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                key_preferences TEXT,  -- JSON string
                turn_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        
        self.connection.commit()
        logger.info("Conversation memory database initialized")
    
    def add_conversation_turn(self, user_id: str, session_id: str, turn: ConversationTurn):
        """Add a conversation turn to memory"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO conversations (
                user_id, session_id, timestamp, user_message, assistant_response,
                intent, entities, filters, recommendations, user_feedback, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, session_id, turn.timestamp, turn.user_message, turn.assistant_response,
            turn.intent, json.dumps(turn.entities), json.dumps(turn.filters),
            json.dumps(turn.recommendations), turn.user_feedback, turn.confidence
        ))
        
        self.connection.commit()
        
        # Update user profile
        self._update_user_profile(user_id, turn)
    
    def get_conversation_history(self, user_id: str, session_id: Optional[str] = None,
                               limit: int = 10) -> List[ConversationTurn]:
        """Get conversation history for a user"""
        cursor = self.connection.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, session_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
        
        rows = cursor.fetchall()
        
        turns = []
        for row in rows:
            turn = ConversationTurn(
                timestamp=row['timestamp'],
                user_message=row['user_message'],
                assistant_response=row['assistant_response'],
                intent=row['intent'] or '',
                entities=json.loads(row['entities'] or '{}'),
                filters=json.loads(row['filters'] or '{}'),
                recommendations=json.loads(row['recommendations'] or '[]'),
                user_feedback=row['user_feedback'],
                confidence=row['confidence'] or 0.0
            )
            turns.append(turn)
        
        return list(reversed(turns))  # Return in chronological order
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with learned preferences"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT * FROM user_profiles WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return UserProfile(
            user_id=user_id,
            preferred_themes=json.loads(row['preferred_themes'] or '{}'),
            preferred_complexity=row['preferred_complexity'] or 'intermediate',
            typical_piece_range=(row['min_pieces'] or 0, row['max_pieces'] or 9999),
            budget_range=(row['min_price'] or 0.0, row['max_price'] or 999.99),
            favorite_recipients=json.loads(row['favorite_recipients'] or '[]'),
            building_style=row['building_style'] or 'general',
            interaction_count=row['interaction_count'] or 0,
            last_active=row['last_active']
        )
    
    def _update_user_profile(self, user_id: str, turn: ConversationTurn):
        """Update user profile based on conversation turn"""
        cursor = self.connection.cursor()
        
        # Get existing profile or create new one
        profile = self.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(
                user_id=user_id,
                preferred_themes={},
                preferred_complexity='intermediate',
                typical_piece_range=(0, 9999),
                budget_range=(0.0, 999.99),
                favorite_recipients=[],
                building_style='general'
            )
        
        # Update based on current turn
        self._learn_from_turn(profile, turn)
        
        # Save updated profile
        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles (
                user_id, preferred_themes, preferred_complexity,
                min_pieces, max_pieces, min_price, max_price,
                favorite_recipients, building_style, interaction_count, last_active, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            json.dumps(profile.preferred_themes),
            profile.preferred_complexity,
            profile.typical_piece_range[0],
            profile.typical_piece_range[1],
            profile.budget_range[0],
            profile.budget_range[1],
            json.dumps(profile.favorite_recipients),
            profile.building_style,
            profile.interaction_count + 1,
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        self.connection.commit()
    
    def _learn_from_turn(self, profile: UserProfile, turn: ConversationTurn):
        """Learn user preferences from a conversation turn"""
        
        # Learn theme preferences
        if turn.filters.get('themes'):
            for theme in turn.filters['themes']:
                current_score = profile.preferred_themes.get(theme, 0.0)
                # Positive feedback boosts preference
                boost = 1.0 if turn.user_feedback == 'liked' else 0.1
                profile.preferred_themes[theme] = current_score + boost
        
        # Learn from recommended sets
        for rec in turn.recommendations:
            theme = rec.get('theme')
            if theme:
                current_score = profile.preferred_themes.get(theme, 0.0)
                # Small boost for viewing recommendations
                profile.preferred_themes[theme] = current_score + 0.05
        
        # Learn complexity preference
        if turn.entities.get('complexity'):
            profile.preferred_complexity = turn.entities['complexity']
        
        # Learn piece count preferences
        if turn.filters.get('min_pieces') and turn.filters.get('max_pieces'):
            min_pieces = turn.filters['min_pieces']
            max_pieces = turn.filters['max_pieces']
            
            # Weighted average with existing preferences
            current_min, current_max = profile.typical_piece_range
            profile.typical_piece_range = (
                int((current_min + min_pieces) / 2),
                int((current_max + max_pieces) / 2)
            )
        
        # Learn budget preferences
        if turn.filters.get('price_range'):
            min_price, max_price = turn.filters['price_range']
            current_min, current_max = profile.budget_range
            profile.budget_range = (
                (current_min + min_price) / 2,
                (current_max + max_price) / 2
            )
        
        # Learn recipient preferences
        if turn.entities.get('recipient'):
            recipient = turn.entities['recipient']
            if recipient not in profile.favorite_recipients:
                profile.favorite_recipients.append(recipient)
    
    def generate_conversation_summary(self, user_id: str, session_id: str) -> str:
        """Generate a summary of the conversation using HuggingFace models"""
        
        # Get conversation history
        history = self.get_conversation_history(user_id, session_id)
        
        if not history:
            return "No conversation history available."
        
        # Create text for summarization
        conversation_text = []
        for turn in history:
            conversation_text.append(f"User: {turn.user_message}")
            conversation_text.append(f"Assistant: {turn.assistant_response}")
        
        full_text = "\n".join(conversation_text)
        
        # Use HuggingFace summarizer if available
        if self.summarizer and len(full_text) > 100:
            try:
                # Truncate if too long for model
                max_length = 1024  # BART max input length
                if len(full_text) > max_length:
                    full_text = full_text[:max_length]
                
                summary = self.summarizer(
                    full_text,
                    max_length=130,
                    min_length=30,
                    do_sample=False
                )
                
                summary_text = summary[0]['summary_text']
                
                # Save summary to database
                self._save_conversation_summary(user_id, session_id, summary_text, history)
                
                return summary_text
                
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        
        # Fallback to rule-based summary
        return self._generate_rule_based_summary(history)
    
    def _generate_rule_based_summary(self, history: List[ConversationTurn]) -> str:
        """Generate a rule-based summary as fallback"""
        if not history:
            return "No conversation history."
        
        # Extract key information
        themes_mentioned = set()
        intents_seen = set()
        total_recommendations = 0
        
        for turn in history:
            if turn.intent:
                intents_seen.add(turn.intent)
            
            for theme in turn.filters.get('themes', []):
                themes_mentioned.add(theme)
            
            total_recommendations += len(turn.recommendations)
        
        # Build summary
        summary_parts = []
        summary_parts.append(f"Conversation with {len(history)} exchanges.")
        
        if themes_mentioned:
            summary_parts.append(f"Discussed themes: {', '.join(themes_mentioned)}.")
        
        if total_recommendations > 0:
            summary_parts.append(f"Provided {total_recommendations} recommendations.")
        
        main_intent = max(intents_seen, key=lambda x: sum(1 for turn in history if turn.intent == x)) if intents_seen else 'general'
        summary_parts.append(f"Primary focus: {main_intent}.")
        
        return " ".join(summary_parts)
    
    def _save_conversation_summary(self, user_id: str, session_id: str, 
                                 summary_text: str, history: List[ConversationTurn]):
        """Save conversation summary to database"""
        cursor = self.connection.cursor()
        
        # Extract key preferences
        key_preferences = {
            'themes': list(set(theme for turn in history for theme in turn.filters.get('themes', []))),
            'intents': list(set(turn.intent for turn in history if turn.intent)),
            'recipients': list(set(turn.entities.get('recipient') for turn in history if turn.entities.get('recipient')))
        }
        
        cursor.execute("""
            INSERT INTO conversation_summaries (
                user_id, session_id, summary_text, key_preferences, turn_count
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            user_id, session_id, summary_text, 
            json.dumps(key_preferences), len(history)
        ))
        
        self.connection.commit()
    
    def get_contextual_suggestions(self, user_id: str, current_query: str) -> List[str]:
        """Get contextual follow-up suggestions based on conversation history"""
        
        profile = self.get_user_profile(user_id)
        recent_history = self.get_conversation_history(user_id, limit=5)
        
        suggestions = []
        
        # Based on user profile
        if profile:
            if profile.preferred_themes:
                top_theme = max(profile.preferred_themes.items(), key=lambda x: x[1])[0]
                suggestions.append(f"Show me more {top_theme} sets")
            
            if profile.favorite_recipients:
                recipient = profile.favorite_recipients[0]
                suggestions.append(f"Find something for my {recipient}")
        
        # Based on recent conversation
        if recent_history:
            last_turn = recent_history[-1]
            
            if last_turn.intent == 'search':
                suggestions.append("Show me similar sets to those")
                suggestions.append("What about something more challenging?")
            
            elif last_turn.intent == 'gift_recommendation':
                suggestions.append("Any alternatives in a different theme?")
                suggestions.append("Something with a smaller piece count?")
            
            # Theme-based suggestions
            recent_themes = set()
            for turn in recent_history:
                recent_themes.update(turn.filters.get('themes', []))
            
            if recent_themes:
                for theme in recent_themes:
                    suggestions.append(f"More {theme} options?")
        
        # Current query context
        current_lower = current_query.lower()
        if 'similar' in current_lower:
            suggestions.append("Show me sets with different piece counts")
            suggestions.append("What about from other themes?")
        
        # Return unique suggestions
        return list(set(suggestions))[:5]  # Limit to 5 suggestions
    
    def get_personalization_context(self, user_id: str) -> Dict[str, Any]:
        """Get personalization context for enhancing recommendations"""
        
        profile = self.get_user_profile(user_id)
        recent_history = self.get_conversation_history(user_id, limit=10)
        
        context = {
            'user_profile': asdict(profile) if profile else {},
            'recent_themes': [],
            'recent_intents': [],
            'interaction_patterns': {},
            'preference_strength': 'low'
        }
        
        if recent_history:
            # Extract recent patterns
            theme_counts = Counter()
            intent_counts = Counter()
            
            for turn in recent_history:
                for theme in turn.filters.get('themes', []):
                    theme_counts[theme] += 1
                
                if turn.intent:
                    intent_counts[turn.intent] += 1
            
            context['recent_themes'] = [theme for theme, count in theme_counts.most_common(3)]
            context['recent_intents'] = [intent for intent, count in intent_counts.most_common(3)]
            
            # Determine preference strength
            if profile and profile.interaction_count > 10:
                context['preference_strength'] = 'high'
            elif profile and profile.interaction_count > 3:
                context['preference_strength'] = 'medium'
        
        return context
    
    def record_user_feedback(self, user_id: str, session_id: str, 
                           conversation_index: int, feedback: str):
        """Record user feedback for a specific conversation turn"""
        cursor = self.connection.cursor()
        
        # Find the conversation turn
        history = self.get_conversation_history(user_id, session_id)
        if conversation_index < len(history):
            turn_timestamp = history[conversation_index].timestamp
            
            cursor.execute("""
                UPDATE conversations 
                SET user_feedback = ?
                WHERE user_id = ? AND session_id = ? AND timestamp = ?
            """, (feedback, user_id, session_id, turn_timestamp))
            
            self.connection.commit()
            
            # Re-learn from updated feedback
            updated_turn = history[conversation_index]
            updated_turn.user_feedback = feedback
            self._update_user_profile(user_id, updated_turn)
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up old conversation data"""
        cursor = self.connection.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        cursor.execute("""
            DELETE FROM conversations 
            WHERE timestamp < ?
        """, (cutoff_date,))
        
        cursor.execute("""
            DELETE FROM conversation_summaries 
            WHERE created_at < ?
        """, (cutoff_date,))
        
        self.connection.commit()
        logger.info(f"Cleaned up conversations older than {days_old} days")
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        profile = self.get_user_profile(user_id)
        history = self.get_conversation_history(user_id, limit=1000)
        
        return {
            'user_id': user_id,
            'profile': asdict(profile) if profile else {},
            'conversation_history': [asdict(turn) for turn in history],
            'export_timestamp': datetime.now().isoformat()
        }
    
    def delete_user_data(self, user_id: str):
        """Delete all user data for GDPR compliance"""
        cursor = self.connection.cursor()
        
        cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM conversation_summaries WHERE user_id = ?", (user_id,))
        
        self.connection.commit()
        logger.info(f"Deleted all data for user {user_id}")
    
    def get_conversation_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics about conversations"""
        cursor = self.connection.cursor()
        
        if user_id:
            # User-specific analytics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(confidence) as avg_confidence,
                    intent,
                    COUNT(*) as intent_count
                FROM conversations 
                WHERE user_id = ?
                GROUP BY intent
            """, (user_id,))
        else:
            # Global analytics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(confidence) as avg_confidence,
                    intent,
                    COUNT(*) as intent_count
                FROM conversations 
                GROUP BY intent
            """)
        
        rows = cursor.fetchall()
        
        analytics = {
            'conversation_stats': {},
            'intent_distribution': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for row in rows:
            if 'total_conversations' in row.keys():
                analytics['conversation_stats'].update(dict(row))
            
            if row['intent']:
                analytics['intent_distribution'][row['intent']] = row['intent_count']
        
        return analytics
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Conversation memory database connection closed")

class EnhancedConversationHandler:
    """Enhanced conversation handler that integrates with HuggingFace NLP"""
    
    def __init__(self, memory_db: ConversationMemoryDB, nlp_recommender):
        """
        Initialize enhanced conversation handler
        
        Args:
            memory_db: ConversationMemoryDB instance
            nlp_recommender: HuggingFaceNLPRecommender instance
        """
        self.memory_db = memory_db
        self.nlp_recommender = nlp_recommender
    
    def process_conversation_turn(self, user_id: str, session_id: str, 
                                user_message: str) -> Dict[str, Any]:
        """
        Process a complete conversation turn with memory integration
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            
        Returns:
            Complete conversation response with recommendations
        """
        
        # Get user profile and personalization context
        personalization_context = self.memory_db.get_personalization_context(user_id)
        
        # Process the natural language query
        processed_query = self.nlp_recommender.process_natural_language_query(
            user_message, user_id, use_conversation_context=True
        )
        
        # Enhance query with personalization
        enhanced_query = self._enhance_with_personalization(processed_query, personalization_context)
        
        # Get recommendations
        recommendations = self.nlp_recommender.search_recommendations(enhanced_query)
        
        # Generate conversational response
        conversation_context = self.memory_db.get_conversation_history(user_id, session_id, limit=3)
        context_text = " ".join([turn.user_message for turn in conversation_context[-2:]])
        
        assistant_response = self.nlp_recommender.generate_conversational_response(
            user_message, context_text, recommendations
        )
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_message=user_message,
            assistant_response=assistant_response,
            intent=enhanced_query['intent'],
            entities=enhanced_query['entities'],
            filters=enhanced_query['filters'],
            recommendations=recommendations,
            confidence=enhanced_query['confidence']
        )
        
        # Save to memory
        self.memory_db.add_conversation_turn(user_id, session_id, turn)
        
        # Get follow-up suggestions
        follow_up_suggestions = self.memory_db.get_contextual_suggestions(user_id, user_message)
        
        return {
            'response': assistant_response,
            'recommendations': recommendations,
            'intent': enhanced_query['intent'],
            'confidence': enhanced_query['confidence'],
            'follow_up_suggestions': follow_up_suggestions,
            'conversation_context': asdict(turn)
        }
    
    def _enhance_with_personalization(self, processed_query: Dict[str, Any], 
                                    personalization_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance processed query with personalization context"""
        
        enhanced_query = processed_query.copy()
        user_profile = personalization_context.get('user_profile', {})
        
        # Enhance theme preferences
        if user_profile.get('preferred_themes') and not enhanced_query.get('filters', {}).get('themes'):
            # Add user's preferred themes as suggestions
            top_themes = sorted(
                user_profile['preferred_themes'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]  # Top 2 themes
            
            if top_themes:
                enhanced_query.setdefault('filters', {})['suggested_themes'] = [theme for theme, _ in top_themes]
        
        # Enhance piece count preferences
        if user_profile.get('typical_piece_range') and not any(
            key in enhanced_query.get('filters', {}) for key in ['min_pieces', 'max_pieces']
        ):
            min_pieces, max_pieces = user_profile['typical_piece_range']
            if min_pieces > 0 and max_pieces < 9999:
                enhanced_query.setdefault('filters', {}).update({
                    'suggested_min_pieces': min_pieces,
                    'suggested_max_pieces': max_pieces
                })
        
        # Boost confidence for personalized queries
        if personalization_context.get('preference_strength') == 'high':
            enhanced_query['confidence'] = min(enhanced_query['confidence'] + 0.2, 1.0)
        elif personalization_context.get('preference_strength') == 'medium':
            enhanced_query['confidence'] = min(enhanced_query['confidence'] + 0.1, 1.0)
        
        return enhanced_query
    
    def handle_follow_up_query(self, user_id: str, session_id: str, 
                             follow_up_message: str) -> Dict[str, Any]:
        """Handle follow-up queries with enhanced context awareness"""
        
        # Get recent conversation context
        recent_history = self.memory_db.get_conversation_history(user_id, session_id, limit=3)
        
        if recent_history:
            last_turn = recent_history[-1]
            
            # Check if this is a follow-up reference
            follow_up_indicators = ['that', 'those', 'similar', 'like that', 'alternative']
            is_follow_up = any(indicator in follow_up_message.lower() for indicator in follow_up_indicators)
            
            if is_follow_up:
                # Enhance the follow-up message with context from previous turn
                enhanced_message = f"{follow_up_message} (based on previous interest in {last_turn.intent}"
                if last_turn.filters.get('themes'):
                    enhanced_message += f" from {', '.join(last_turn.filters['themes'])} themes"
                enhanced_message += ")"
                
                return self.process_conversation_turn(user_id, session_id, enhanced_message)
        
        # Process as regular conversation turn
        return self.process_conversation_turn(user_id, session_id, follow_up_message)
    
    def get_conversation_summary(self, user_id: str, session_id: str) -> str:
        """Get a summary of the conversation"""
        return self.memory_db.generate_conversation_summary(user_id, session_id)
    
    def record_feedback(self, user_id: str, session_id: str, 
                       turn_index: int, feedback: str):
        """Record user feedback for learning"""
        self.memory_db.record_user_feedback(user_id, session_id, turn_index, feedback)
    
    def clear_user_memory(self, user_id: str):
        """Clear conversation memory for a user"""
        self.memory_db.delete_user_data(user_id)
