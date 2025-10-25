import json
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime

class VectorService:
    """Service for handling vector embeddings and similarity search"""
    
    def __init__(self):
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.id_to_text = {}  # Map FAISS IDs to original text
        self.id_to_metadata = {}  # Map FAISS IDs to metadata
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index from disk"""
        try:
            if os.path.exists('vector_index.faiss'):
                self.index = faiss.read_index('vector_index.faiss')
                with open('vector_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_text = metadata['id_to_text']
                    self.id_to_metadata = metadata['id_to_metadata']
                print(f"✅ Loaded vector index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}")
            # Start with empty index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, 'vector_index.faiss')
            metadata = {
                'id_to_text': self.id_to_text,
                'id_to_metadata': self.id_to_metadata
            }
            with open('vector_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            print(f"✅ Saved vector index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Error saving index: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def add_goals_to_index(self, goals: List[Dict[str, Any]]) -> List[int]:
        """Add city goals to the vector index"""
        texts = []
        metadata_list = []
        
        for goal in goals:
            # Create searchable text from goal
            text = f"{goal['goal_title']} {goal['goal_description']} {goal['target_metric']}"
            texts.append(text)
            metadata_list.append({
                'type': 'city_goal',
                'city_name': goal['city_name'],
                'goal_id': goal.get('id'),
                'priority_level': goal.get('priority_level', 'medium'),
                'original_text': text
            })
        
        # Create embeddings
        embeddings = self.create_embeddings_batch(texts)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            vector_id = start_id + i
            self.id_to_text[vector_id] = text
            self.id_to_metadata[vector_id] = metadata
        
        self._save_index()
        return list(range(start_id, start_id + len(goals)))
    
    def add_policies_to_index(self, policies: List[Dict[str, Any]]) -> List[int]:
        """Add policy documents to the vector index"""
        texts = []
        metadata_list = []
        
        for policy in policies:
            # Create searchable text from policy
            text = f"{policy['title']} {policy['content']}"
            texts.append(text)
            metadata_list.append({
                'type': 'policy_document',
                'source': policy['source'],
                'document_type': policy['document_type'],
                'geographic_scope': policy['geographic_scope'],
                'topic_tags': policy.get('topic_tags', []),
                'policy_id': policy.get('id'),
                'original_text': text
            })
        
        # Create embeddings
        embeddings = self.create_embeddings_batch(texts)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        # Store metadata
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            vector_id = start_id + i
            self.id_to_text[vector_id] = text
            self.id_to_metadata[vector_id] = metadata
        
        self._save_index()
        return list(range(start_id, start_id + len(policies)))
    
    def search_similar(self, query_text: str, k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """Search for similar texts in the vector index"""
        if self.index.ntotal == 0:
            return []
        
        # Create query embedding
        query_embedding = self.create_embedding(query_text)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            metadata = self.id_to_metadata.get(idx, {})
            
            # Apply filter if specified
            if filter_type and metadata.get('type') != filter_type:
                continue
            
            results.append({
                'vector_id': int(idx),
                'similarity_score': float(score),
                'text': self.id_to_text.get(idx, ''),
                'metadata': metadata
            })
        
        return results
    
    def get_goal_aligned_recommendations(self, city_name: str, problem_description: str, 
                                       current_data: Dict[str, Any], max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get goal-aligned recommendations using RAG"""
        
        # Create search query combining problem and city context
        search_query = f"City: {city_name}. Problem: {problem_description}. Data: {json.dumps(current_data, default=str)}"
        
        # Search for relevant goals first
        goal_results = self.search_similar(search_query, k=max_recommendations, filter_type='city_goal')
        
        # Filter by city name
        city_goals = [r for r in goal_results if r['metadata'].get('city_name') == city_name]
        
        # Search for relevant policies
        policy_results = self.search_similar(search_query, k=max_recommendations, filter_type='policy_document')
        
        # Combine and rank results
        recommendations = []
        
        for goal in city_goals:
            for policy in policy_results:
                # Calculate combined relevance score
                combined_score = (goal['similarity_score'] + policy['similarity_score']) / 2
                
                recommendation = {
                    'city_goal': goal,
                    'policy_document': policy,
                    'combined_score': combined_score,
                    'recommendation_type': 'goal_aligned'
                }
                recommendations.append(recommendation)
        
        # Sort by combined score and return top results
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return recommendations[:max_recommendations]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'goals_count': len([m for m in self.id_to_metadata.values() if m.get('type') == 'city_goal']),
            'policies_count': len([m for m in self.id_to_metadata.values() if m.get('type') == 'policy_document']),
            'cities_with_goals': len(set(m.get('city_name', '') for m in self.id_to_metadata.values() 
                                       if m.get('type') == 'city_goal' and m.get('city_name')))
        }

