"""
FastAPI Integration for HuggingFace-based LEGO NLP Recommender

This module provides FastAPI endpoints that integrate with the new HuggingFace-based
NLP system, replacing the Ollama dependency while maintaining API compatibility.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uuid

from hf_nlp_recommender import HuggingFaceNLPRecommender
from hf_conversation_memory import ConversationMemoryDB, EnhancedConversationHandler

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class NaturalLanguageQuery(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query about LEGO sets")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    top_k: int = Field(10, description="Number of recommendations to return")
    use_context: bool = Field(True, description="Whether to use conversation context")

class ConversationalMessage(BaseModel):
    """Conversational message request"""
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    include_suggestions: bool = Field(True, description="Include follow-up suggestions")

class UserFeedback(BaseModel):
    """User feedback request"""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    turn_index: int = Field(..., description="Index of conversation turn")
    feedback: str = Field(..., description="Feedback type: 'liked', 'disliked', 'helpful', 'not_helpful'")
    rating: Optional[int] = Field(None, description="Optional rating 1-5")

class QueryUnderstandingRequest(BaseModel):
    """Query understanding analysis request"""
    query: str = Field(..., description="Query to analyze")
    include_explanation: bool = Field(True, description="Include explanation of processing")

# Response models
class NLProcessingResult(BaseModel):
    """Natural language processing result"""
    query: str
    intent: str
    entities: Dict[str, Any]
    filters: Dict[str, Any] 
    semantic_query: str
    confidence: float
    explanation: Optional[str] = None

class ConversationResponse(BaseModel):
    """Conversational AI response"""
    response: str
    recommendations: List[Dict[str, Any]]
    intent: str
    confidence: float
    follow_up_suggestions: List[str]
    session_id: str
    conversation_context: Dict[str, Any]

class HuggingFaceNLPAPI:
    """FastAPI integration for HuggingFace NLP Recommender"""
    
    def __init__(self, dbcon, memory_db_path: str = "conversation_memory.db"):
        """
        Initialize the HuggingFace NLP API
        
        Args:
            dbcon: Database connection for LEGO data
            memory_db_path: Path to conversation memory database
        """
        self.dbcon = dbcon
        
        # Initialize HuggingFace NLP Recommender
        self.nlp_recommender = HuggingFaceNLPRecommender(dbcon)
        
        # Initialize conversation memory
        self.memory_db = ConversationMemoryDB(memory_db_path)
        
        # Initialize enhanced conversation handler
        self.conversation_handler = EnhancedConversationHandler(
            self.memory_db, self.nlp_recommender
        )
        
        # Track active sessions
        self.active_sessions = {}
        
        logger.info("HuggingFace NLP API initialized successfully")
    
    def process_natural_language_query(self, request: NaturalLanguageQuery) -> Dict[str, Any]:
        """
        Process natural language query and return recommendations
        
        Args:
            request: NaturalLanguageQuery request
            
        Returns:
            Dictionary with processed results and recommendations
        """
        try:
            # Process the query
            processed_query = self.nlp_recommender.process_natural_language_query(
                request.query, 
                request.user_id, 
                request.use_context
            )
            
            # Get recommendations
            recommendations = self.nlp_recommender.search_recommendations(
                processed_query, 
                request.top_k
            )
            
            # Add to conversation memory if user_id provided
            if request.user_id:
                session_id = f"search_{uuid.uuid4().hex[:8]}"
                assistant_response = f"Found {len(recommendations)} LEGO sets matching your criteria."
                
                self.nlp_recommender.add_conversation_interaction(
                    request.query, assistant_response, recommendations
                )
            
            return {
                'query': request.query,
                'intent': processed_query['intent'],
                'entities': processed_query['entities'],
                'filters': processed_query['filters'],
                'confidence': processed_query['confidence'],
                'recommendations': recommendations,
                'total_found': len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Natural language query processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    def handle_conversational_message(self, request: ConversationalMessage) -> ConversationResponse:
        """
        Handle conversational message with full context awareness
        
        Args:
            request: ConversationalMessage request
            
        Returns:
            ConversationResponse with AI response and recommendations
        """
        try:
            # Generate session ID if not provided
            session_id = request.session_id or f"conv_{uuid.uuid4().hex[:12]}"
            
            # Process conversation turn
            result = self.conversation_handler.process_conversation_turn(
                request.user_id, session_id, request.message
            )
            
            # Track active session
            self.active_sessions[session_id] = {
                'user_id': request.user_id,
                'last_activity': datetime.now().isoformat()
            }
            
            return ConversationResponse(
                response=result['response'],
                recommendations=result['recommendations'],
                intent=result['intent'],
                confidence=result['confidence'],
                follow_up_suggestions=result['follow_up_suggestions'] if request.include_suggestions else [],
                session_id=session_id,
                conversation_context=result['conversation_context']
            )
            
        except Exception as e:
            logger.error(f"Conversational message handling failed: {e}")
            raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")
    
    def handle_follow_up_message(self, user_id: str, session_id: str, message: str) -> ConversationResponse:
        """
        Handle follow-up message with enhanced context awareness
        
        Args:
            user_id: User ID
            session_id: Session ID
            message: Follow-up message
            
        Returns:
            ConversationResponse with contextual response
        """
        try:
            result = self.conversation_handler.handle_follow_up_query(
                user_id, session_id, message
            )
            
            return ConversationResponse(
                response=result['response'],
                recommendations=result['recommendations'],
                intent=result['intent'],
                confidence=result['confidence'],
                follow_up_suggestions=result['follow_up_suggestions'],
                session_id=session_id,
                conversation_context=result['conversation_context']
            )
            
        except Exception as e:
            logger.error(f"Follow-up message handling failed: {e}")
            raise HTTPException(status_code=500, detail=f"Follow-up processing failed: {str(e)}")
    
    def understand_query(self, request: QueryUnderstandingRequest) -> NLProcessingResult:
        """
        Analyze and explain query understanding
        
        Args:
            request: QueryUnderstandingRequest
            
        Returns:
            NLProcessingResult with detailed analysis
        """
        try:
            # Process the query
            processed_query = self.nlp_recommender.process_natural_language_query(
                request.query, use_conversation_context=False
            )
            
            explanation = None
            if request.include_explanation:
                explanation = self._generate_processing_explanation(processed_query)
            
            return NLProcessingResult(
                query=request.query,
                intent=processed_query['intent'],
                entities=processed_query['entities'],
                filters=processed_query['filters'],
                semantic_query=processed_query['semantic_query'],
                confidence=processed_query['confidence'],
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")
    
    def _generate_processing_explanation(self, processed_query: Dict[str, Any]) -> str:
        """Generate explanation of query processing"""
        explanation_parts = []
        
        explanation_parts.append(f"**Intent Detection**: Classified as '{processed_query['intent']}'")
        
        if processed_query['entities']:
            entity_descriptions = []
            for key, value in processed_query['entities'].items():
                entity_descriptions.append(f"{key}: {value}")
            explanation_parts.append(f"**Entities Extracted**: {', '.join(entity_descriptions)}")
        
        if processed_query['filters']:
            filter_descriptions = []
            for key, value in processed_query['filters'].items():
                if isinstance(value, list):
                    filter_descriptions.append(f"{key}: {', '.join(map(str, value))}")
                else:
                    filter_descriptions.append(f"{key}: {value}")
            explanation_parts.append(f"**Filters Applied**: {', '.join(filter_descriptions)}")
        
        explanation_parts.append(f"**Confidence Score**: {processed_query['confidence']:.2f}")
        explanation_parts.append(f"**Semantic Query**: \"{processed_query['semantic_query']}\"")
        
        return "\n\n".join(explanation_parts)
    
    def record_user_feedback(self, request: UserFeedback) -> Dict[str, str]:
        """
        Record user feedback for learning
        
        Args:
            request: UserFeedback request
            
        Returns:
            Confirmation message
        """
        try:
            self.conversation_handler.record_feedback(
                request.user_id, request.session_id, request.turn_index, request.feedback
            )
            
            return {"message": f"Feedback '{request.feedback}' recorded successfully"}
            
        except Exception as e:
            logger.error(f"Feedback recording failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feedback recording failed: {str(e)}")
    
    def get_conversation_memory(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get conversation memory for a user
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            
        Returns:
            Conversation memory data
        """
        try:
            history = self.memory_db.get_conversation_history(user_id, session_id)
            profile = self.memory_db.get_user_profile(user_id)
            
            return {
                'user_id': user_id,
                'session_id': session_id,
                'conversation_history': [
                    {
                        'timestamp': turn.timestamp,
                        'user_message': turn.user_message,
                        'assistant_response': turn.assistant_response,
                        'intent': turn.intent,
                        'confidence': turn.confidence
                    }
                    for turn in history
                ],
                'user_profile': profile.__dict__ if profile else None,
                'total_interactions': len(history)
            }
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Memory retrieval failed: {str(e)}")
    
    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, str]:
        """
        Get conversation summary
        
        Args:
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Conversation summary
        """
        try:
            summary = self.conversation_handler.get_conversation_summary(user_id, session_id)
            
            return {
                'user_id': user_id,
                'session_id': session_id,
                'summary': summary,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")
    
    def clear_user_memory(self, user_id: str) -> Dict[str, str]:
        """
        Clear user conversation memory
        
        Args:
            user_id: User ID
            
        Returns:
            Confirmation message
        """
        try:
            self.conversation_handler.clear_user_memory(user_id)
            
            return {"message": f"Memory cleared for user {user_id}"}
            
        except Exception as e:
            logger.error(f"Memory clearing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Memory clearing failed: {str(e)}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get health status of the HuggingFace NLP system
        
        Returns:
            System health status
        """
        try:
            nlp_health = self.nlp_recommender.get_health_status()
            memory_analytics = self.memory_db.get_conversation_analytics()
            
            return {
                'nlp_system': nlp_health,
                'conversation_memory': {
                    'status': 'healthy',
                    'total_conversations': memory_analytics.get('conversation_stats', {}).get('total_conversations', 0),
                    'active_sessions': len(self.active_sessions)
                },
                'system_status': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'nlp_system': {'status': 'error', 'error': str(e)},
                'conversation_memory': {'status': 'error'},
                'system_status': 'degraded',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific user
        
        Args:
            user_id: User ID
            
        Returns:
            User analytics data
        """
        try:
            analytics = self.memory_db.get_conversation_analytics(user_id)
            personalization_context = self.memory_db.get_personalization_context(user_id)
            
            return {
                'user_id': user_id,
                'analytics': analytics,
                'personalization_context': personalization_context,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"User analytics failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export user data for GDPR compliance
        
        Args:
            user_id: User ID
            
        Returns:
            Complete user data export
        """
        try:
            return self.memory_db.export_user_data(user_id)
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}")
    
    def delete_user_data(self, user_id: str) -> Dict[str, str]:
        """
        Delete user data for GDPR compliance
        
        Args:
            user_id: User ID
            
        Returns:
            Confirmation message
        """
        try:
            self.memory_db.delete_user_data(user_id)
            
            # Remove from active sessions
            sessions_to_remove = [sid for sid, data in self.active_sessions.items() 
                                if data['user_id'] == user_id]
            for sid in sessions_to_remove:
                del self.active_sessions[sid]
            
            return {"message": f"All data deleted for user {user_id}"}
            
        except Exception as e:
            logger.error(f"Data deletion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Data deletion failed: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.memory_db.cleanup_old_conversations()
            self.memory_db.close()
            logger.info("HuggingFace NLP API cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# FastAPI route handlers (to be integrated into main API)
def create_hf_nlp_routes(app: FastAPI, hf_api: HuggingFaceNLPAPI):
    """
    Create FastAPI routes for HuggingFace NLP functionality
    
    Args:
        app: FastAPI application instance
        hf_api: HuggingFaceNLPAPI instance
    """
    
    @app.post("/nlp/query", response_model=Dict[str, Any])
    async def process_natural_language_query(request: NaturalLanguageQuery):
        """Process natural language query and return recommendations"""
        return hf_api.process_natural_language_query(request)
    
    @app.post("/nlp/chat", response_model=ConversationResponse) 
    async def conversational_chat(request: ConversationalMessage):
        """Handle conversational AI chat with memory"""
        return hf_api.handle_conversational_message(request)
    
    @app.post("/nlp/follow-up")
    async def handle_follow_up(
        user_id: str = Query(..., description="User ID"),
        session_id: str = Query(..., description="Session ID"),
        message: str = Query(..., description="Follow-up message")
    ):
        """Handle follow-up message with context"""
        return hf_api.handle_follow_up_message(user_id, session_id, message)
    
    @app.post("/nlp/understand", response_model=NLProcessingResult)
    async def understand_query(request: QueryUnderstandingRequest):
        """Analyze and explain query understanding"""
        return hf_api.understand_query(request)
    
    @app.post("/nlp/feedback")
    async def record_feedback(request: UserFeedback):
        """Record user feedback for learning"""
        return hf_api.record_user_feedback(request)
    
    @app.get("/nlp/memory/{user_id}")
    async def get_conversation_memory(
        user_id: str,
        session_id: Optional[str] = Query(None, description="Optional session ID")
    ):
        """Get conversation memory for user"""
        return hf_api.get_conversation_memory(user_id, session_id)
    
    @app.get("/nlp/summary/{user_id}/{session_id}")
    async def get_conversation_summary(user_id: str, session_id: str):
        """Get conversation summary"""
        return hf_api.get_conversation_summary(user_id, session_id)
    
    @app.delete("/nlp/memory/{user_id}")
    async def clear_user_memory(user_id: str):
        """Clear user conversation memory"""
        return hf_api.clear_user_memory(user_id)
    
    @app.get("/nlp/health")
    async def get_nlp_health():
        """Get HuggingFace NLP system health"""
        return hf_api.get_system_health()
    
    @app.get("/nlp/analytics/{user_id}")
    async def get_user_analytics(user_id: str):
        """Get user analytics"""
        return hf_api.get_user_analytics(user_id)
    
    @app.get("/nlp/export/{user_id}")
    async def export_user_data(user_id: str):
        """Export user data (GDPR)"""
        return hf_api.export_user_data(user_id)
    
    @app.delete("/nlp/user/{user_id}")
    async def delete_user_data(user_id: str):
        """Delete user data (GDPR)"""
        return hf_api.delete_user_data(user_id)
