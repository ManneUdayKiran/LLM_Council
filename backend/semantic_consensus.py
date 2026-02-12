"""Semantic Consensus Engine for LLM Council.

Instead of exact text matching, this engine:
1. Converts each model's answer into embeddings
2. Computes similarity matrix
3. Selects highest-agreement answer or generates merged answer
"""

import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import httpx

try:
    from .config import GROQ_API_KEY
    from .openrouter import query_model
except ImportError:
    from config import GROQ_API_KEY
    from openrouter import query_model


@dataclass
class ConsensusResult:
    """Result of semantic consensus analysis."""
    best_answer_model: str
    best_answer_content: str
    consensus_score: float  # 0-1, higher = more agreement
    similarity_matrix: List[List[float]]
    model_consensus_scores: Dict[str, float]
    confidence_metrics: Dict[str, Any]
    merged_answer: Optional[str] = None
    hallucination_risk: Optional[Dict[str, Any]] = None


class SemanticConsensusEngine:
    """Engine for computing semantic consensus among LLM responses."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """
        Initialize the consensus engine.
        
        Args:
            embedding_model: Model to use for generating embeddings (Groq compatible)
        """
        self.embedding_model = embedding_model
        self.api_key = GROQ_API_KEY
        
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using simple text-based approach.
        
        Since Groq doesn't support embeddings, we use TF-IDF-like representation.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Simple bag-of-words representation with character n-grams
        # This creates a pseudo-embedding without external API
        words = text.lower().split()
        
        # Create a fixed-size vector (256 dimensions)
        vector = np.zeros(256)
        
        for i, word in enumerate(words[:256]):
            # Use word hash to map to vector position
            hash_val = hash(word) % 256
            vector[hash_val] += 1.0 / (i + 1)  # Weight by position
        
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts in parallel.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        tasks = [self.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def compute_similarity_matrix(
        self, embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            N x N similarity matrix
        """
        n = len(embeddings)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif i < j:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        
        return matrix
    
    def calculate_consensus_scores(
        self, similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate consensus score for each response.
        
        The consensus score is the average similarity to all other responses.
        Higher score = more agreement with other models.
        
        Args:
            similarity_matrix: N x N similarity matrix
            
        Returns:
            Array of consensus scores (one per model)
        """
        n = similarity_matrix.shape[0]
        scores = np.zeros(n)
        
        for i in range(n):
            # Average similarity to all OTHER responses (exclude self)
            other_similarities = [
                similarity_matrix[i][j] for j in range(n) if i != j
            ]
            scores[i] = np.mean(other_similarities) if other_similarities else 0.0
        
        return scores
    
    def calculate_confidence_metrics(
        self, similarity_matrix: np.ndarray, consensus_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate overall confidence metrics for the consensus.
        
        Args:
            similarity_matrix: N x N similarity matrix
            consensus_scores: Array of consensus scores for each model
            
        Returns:
            Dict with confidence, models_agreed, and total_models
        """
        n = len(consensus_scores)
        
        # Calculate average similarity across all pairs
        total_similarity = 0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_similarity += similarity_matrix[i][j]
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        
        # Calculate confidence (0-1 scale)
        # Confidence is based on:
        # 1. Average pairwise similarity (70% weight)
        # 2. Consistency of scores - lower variance = higher confidence (30% weight)
        variance = np.var(consensus_scores)
        consistency_score = 1.0 - min(variance, 1.0)  # Cap at 1.0
        
        confidence = (avg_similarity * 0.7) + (consistency_score * 0.3)
        
        # Count models that agreed (similarity > 0.7 threshold)
        agreement_threshold = 0.7
        models_agreed = sum(
            1 for score in consensus_scores if score >= agreement_threshold
        )
        
        return {
            "confidence": float(confidence),
            "models_agreed": int(models_agreed),
            "total_models": int(n),
        }
    
    async def generate_merged_answer(
        self,
        responses: Dict[str, str],
        user_query: str,
        merger_model: str = "anthropic/claude-sonnet-4.5",
    ) -> str:
        """
        Generate a merged answer that synthesizes all responses.
        
        Args:
            responses: Dict mapping model names to their responses
            user_query: Original user query
            merger_model: Model to use for merging
            
        Returns:
            Merged answer
        """
        # Build context with all responses
        responses_text = "\n\n".join([
            f"Model {i+1} Response:\n{response}"
            for i, response in enumerate(responses.values())
        ])
        
        merge_prompt = f"""You are synthesizing multiple AI model responses into a single, comprehensive answer.

Original Question:
{user_query}

Multiple Model Responses:
{responses_text}

Your task:
1. Identify the common ground and consensus points across responses
2. Include unique valuable insights from any response
3. Resolve any contradictions by noting different perspectives
4. Create a coherent, well-structured answer that represents the collective wisdom

Provide the synthesized answer without meta-commentary about the synthesis process."""

        merged_result = await query_model(
            model=merger_model,
            messages=[{"role": "user", "content": merge_prompt}],
        )
        
        return merged_result["content"] if merged_result else ""
    
    async def find_consensus(
        self,
        responses: Dict[str, str],
        user_query: str,
        generate_merged: bool = True,
        merger_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> ConsensusResult:
        """
        Find semantic consensus among model responses.
        
        Args:
            responses: Dict mapping model names to their responses
            user_query: Original user query (for merged answer generation)
            generate_merged: Whether to generate a merged answer
            merger_model: Model to use for merging
            
        Returns:
            ConsensusResult with analysis and best answer
        """
        model_names = list(responses.keys())
        response_texts = list(responses.values())
        
        # Get embeddings for all responses
        embeddings = await self.get_embeddings_batch(response_texts)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        # Calculate consensus scores
        consensus_scores = self.calculate_consensus_scores(similarity_matrix)
        
        # Calculate confidence metrics
        confidence_metrics = self.calculate_confidence_metrics(
            similarity_matrix, consensus_scores
        )
        
        # Detect disagreement (low confidence threshold)
        disagreement_detected = confidence_metrics["confidence"] < 0.6
        low_consensus = confidence_metrics["models_agreed"] < (len(model_names) / 2)
        
        confidence_metrics["disagreement_detected"] = disagreement_detected
        confidence_metrics["low_consensus"] = low_consensus
        confidence_metrics["warning_message"] = None
        
        if disagreement_detected or low_consensus:
            if confidence_metrics["confidence"] < 0.4:
                confidence_metrics["warning_message"] = "High disagreement: Models have significantly different interpretations. Consider rephrasing your question or providing more context."
            elif confidence_metrics["confidence"] < 0.6:
                confidence_metrics["warning_message"] = "Moderate disagreement: Models have varying perspectives on this topic."
        
        # Find model with highest consensus
        best_idx = int(np.argmax(consensus_scores))
        best_model = model_names[best_idx]
        best_answer = response_texts[best_idx]
        
        # Build model-to-score mapping
        model_consensus_scores = {
            model_names[i]: float(consensus_scores[i])
            for i in range(len(model_names))
        }
        
        # Generate merged answer if requested
        merged_answer = None
        if generate_merged:
            merged_answer = await self.generate_merged_answer(
                responses, user_query, merger_model
            )
        
        # Estimate hallucination risk (lazy import to avoid circular dependency)
        try:
            from .hallucination_detector import estimate_hallucination_risk
        except ImportError:
            from hallucination_detector import estimate_hallucination_risk
        
        hallucination_risk = await estimate_hallucination_risk(
            responses,
            similarity_matrix.tolist(),
            model_consensus_scores
        )
        
        return ConsensusResult(
            best_answer_model=best_model,
            best_answer_content=best_answer,
            consensus_score=float(consensus_scores[best_idx]),
            similarity_matrix=similarity_matrix.tolist(),
            model_consensus_scores=model_consensus_scores,
            confidence_metrics=confidence_metrics,
            merged_answer=merged_answer,
            hallucination_risk=hallucination_risk,
        )


# Global instance
_consensus_engine = SemanticConsensusEngine()


async def analyze_semantic_consensus(
    stage1_results: Dict[str, str],
    user_query: str,
    generate_merged: bool = True,
) -> Dict:
    """
    Analyze semantic consensus among stage1 model responses.
    
    Args:
        stage1_results: Dict mapping model names to their responses
        user_query: Original user query
        generate_merged: Whether to generate a merged answer
        
    Returns:
        Dict with consensus analysis results
    """
    try:
        result = await _consensus_engine.find_consensus(
            stage1_results,
            user_query,
            generate_merged=generate_merged,
        )
        
        return {
            "best_answer": {
                "model": result.best_answer_model,
                "content": result.best_answer_content,
                "consensus_score": result.consensus_score,
            },
            "model_consensus_scores": result.model_consensus_scores,
            "similarity_matrix": result.similarity_matrix,
            "merged_answer": result.merged_answer,
            "final_answer": result.merged_answer or result.best_answer_content,
            "confidence": result.confidence_metrics["confidence"],
            "models_agreed": result.confidence_metrics["models_agreed"],
            "total_models": result.confidence_metrics["total_models"],
            "disagreement_detected": result.confidence_metrics.get("disagreement_detected", False),
            "low_consensus": result.confidence_metrics.get("low_consensus", False),
            "warning_message": result.confidence_metrics.get("warning_message"),
            "hallucination_risk": result.hallucination_risk,
        }
    except Exception as e:
        print(f"Consensus analysis error: {e}")
        # Return a simple fallback based on response length
        model_names = list(stage1_results.keys())
        response_texts = list(stage1_results.values())
        
        # Simple heuristic: pick the middle-length response
        lengths = [(i, len(text)) for i, text in enumerate(response_texts)]
        lengths.sort(key=lambda x: x[1])
        median_idx = lengths[len(lengths) // 2][0]
        
        return {
            "best_answer": {
                "model": model_names[median_idx],
                "content": response_texts[median_idx],
                "consensus_score": 0.5,  # Neutral score
            },
            "model_consensus_scores": {
                model: 0.5 for model in model_names
            },
            "similarity_matrix": [[1.0 if i == j else 0.5 for j in range(len(model_names))] for i in range(len(model_names))],
            "merged_answer": None,
            "final_answer": response_texts[median_idx],
            "confidence": 0.5,
            "models_agreed": len(model_names) // 2,
            "total_models": len(model_names),
            "disagreement_detected": True,
            "low_consensus": True,
            "warning_message": "Unable to analyze consensus due to technical error.",
            "error": str(e),
        }
