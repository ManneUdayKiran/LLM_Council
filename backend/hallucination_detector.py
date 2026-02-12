"""Hallucination Risk Estimator for LLM Council responses."""

import re
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

try:
    from .semantic_consensus import SemanticConsensusEngine
except ImportError:
    from semantic_consensus import SemanticConsensusEngine


class HallucinationRiskEstimator:
    """Estimates the risk of hallucination in LLM responses."""
    
    def __init__(self):
        self.consensus_engine = SemanticConsensusEngine()
    
    def calculate_entropy(self, probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy from probability distribution.
        Higher entropy = more uncertainty/disagreement.
        """
        probabilities = np.array(probabilities)
        # Normalize
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        
        # Shannon entropy: -Σ(p * log(p))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def calculate_variance(self, similarity_scores: List[float]) -> float:
        """
        Calculate variance in similarity scores.
        Higher variance = inconsistent answers.
        """
        if len(similarity_scores) < 2:
            return 0.0
        return float(np.var(similarity_scores))
    
    def detect_contradictions(self, responses: Dict[str, str]) -> Tuple[int, List[str]]:
        """
        Detect contradictions between responses.
        
        Returns:
            Tuple of (contradiction_count, list of contradictions)
        """
        contradictions = []
        response_texts = list(responses.values())
        model_names = list(responses.keys())
        
        # Look for explicit negations
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                text1 = response_texts[i].lower()
                text2 = response_texts[j].lower()
                
                # Check for yes/no contradictions
                if self._contains_affirmative(text1) and self._contains_negative(text2):
                    contradictions.append(
                        f"{model_names[i]} says YES, but {model_names[j]} says NO"
                    )
                elif self._contains_negative(text1) and self._contains_affirmative(text2):
                    contradictions.append(
                        f"{model_names[i]} says NO, but {model_names[j]} says YES"
                    )
                
                # Check for true/false contradictions
                if self._contains_true(text1) and self._contains_false(text2):
                    contradictions.append(
                        f"{model_names[i]} says TRUE, but {model_names[j]} says FALSE"
                    )
                elif self._contains_false(text1) and self._contains_true(text2):
                    contradictions.append(
                        f"{model_names[i]} says FALSE, but {model_names[j]} says TRUE"
                    )
        
        return len(contradictions), contradictions
    
    def _contains_affirmative(self, text: str) -> bool:
        """Check if text contains affirmative statements."""
        patterns = [
            r'\byes\b', r'\bcorrect\b', r'\btrue\b', r'\baffirmative\b',
            r'\bagree\b', r'\bdefinitely\b', r'\bcertainly\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _contains_negative(self, text: str) -> bool:
        """Check if text contains negative statements."""
        patterns = [
            r'\bno\b', r'\bincorrect\b', r'\bfalse\b', r'\bnegative\b',
            r'\bdisagree\b', r'\bnot true\b', r'\bdefinitely not\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _contains_true(self, text: str) -> bool:
        """Check if text explicitly states something is true."""
        patterns = [
            r'\bis true\b', r'\btrue that\b', r'\btruthful\b', r'\baccurate\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _contains_false(self, text: str) -> bool:
        """Check if text explicitly states something is false."""
        patterns = [
            r'\bis false\b', r'\bfalse that\b', r'\buntruthful\b', 
            r'\binaccurate\b', r'\bmisleading\b'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def detect_factual_inconsistencies(self, responses: Dict[str, str]) -> Tuple[int, List[str]]:
        """
        Detect factual inconsistencies (numbers, dates, names that differ).
        
        Returns:
            Tuple of (inconsistency_count, list of inconsistencies)
        """
        inconsistencies = []
        
        # Extract numbers from all responses
        response_texts = list(responses.values())
        model_names = list(responses.keys())
        
        numbers_per_response = []
        for text in response_texts:
            # Extract numbers (including decimals, percentages, years)
            numbers = re.findall(r'\b\d+\.?\d*%?\b', text)
            numbers_per_response.append(set(numbers))
        
        # Check if responses have very different numbers
        if len(numbers_per_response) >= 2:
            all_numbers = set()
            for nums in numbers_per_response:
                all_numbers.update(nums)
            
            if len(all_numbers) > 0:
                # If responses share < 30% of numbers, it's suspicious
                for i in range(len(numbers_per_response)):
                    for j in range(i + 1, len(numbers_per_response)):
                        shared = numbers_per_response[i] & numbers_per_response[j]
                        total = numbers_per_response[i] | numbers_per_response[j]
                        
                        if len(total) > 0:
                            overlap_ratio = len(shared) / len(total)
                            if overlap_ratio < 0.3:
                                inconsistencies.append(
                                    f"{model_names[i]} and {model_names[j]} cite different numbers"
                                )
        
        return len(inconsistencies), inconsistencies
    
    async def estimate_risk(
        self, 
        responses: Dict[str, str],
        similarity_matrix: List[List[float]] = None,
        consensus_scores: Dict[str, float] = None
    ) -> Dict:
        """
        Estimate hallucination risk from responses.
        
        Args:
            responses: Dict mapping model names to their responses
            similarity_matrix: Optional pre-computed similarity matrix
            consensus_scores: Optional pre-computed consensus scores
            
        Returns:
            Dict with risk level, score, and contributing factors
        """
        # 1. Calculate entropy/variance
        if consensus_scores:
            scores = list(consensus_scores.values())
            entropy = self.calculate_entropy(scores)
            variance = self.calculate_variance(scores)
        else:
            entropy = 0.5
            variance = 0.1
        
        # 2. Calculate average similarity
        avg_similarity = 0.5
        if similarity_matrix:
            # Extract upper triangle (exclude diagonal)
            similarities = []
            n = len(similarity_matrix)
            for i in range(n):
                for j in range(i + 1, n):
                    similarities.append(similarity_matrix[i][j])
            
            if similarities:
                avg_similarity = np.mean(similarities)
                variance = self.calculate_variance(similarities)
        
        # 3. Detect contradictions
        contradiction_count, contradictions = self.detect_contradictions(responses)
        
        # 4. Detect factual inconsistencies
        inconsistency_count, inconsistencies = self.detect_factual_inconsistencies(responses)
        
        # Calculate risk score (0-1)
        # Higher entropy = higher risk
        # Lower similarity = higher risk
        # More contradictions = higher risk
        # More inconsistencies = higher risk
        
        entropy_risk = min(entropy / 2.0, 1.0)  # Normalize entropy
        similarity_risk = 1.0 - avg_similarity  # Low similarity = high risk
        variance_risk = min(variance * 2.0, 1.0)  # High variance = high risk
        contradiction_risk = min(contradiction_count / len(responses), 1.0)
        inconsistency_risk = min(inconsistency_count / len(responses), 1.0)
        
        # Weighted average
        risk_score = (
            entropy_risk * 0.25 +
            similarity_risk * 0.30 +
            variance_risk * 0.15 +
            contradiction_risk * 0.20 +
            inconsistency_risk * 0.10
        )
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
            risk_icon = "🔴"
            risk_message = "High risk of hallucination detected. Models show significant disagreement and contradictions."
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
            risk_icon = "🟡"
            risk_message = "Moderate risk of hallucination. Some inconsistencies detected between model responses."
        else:
            risk_level = "LOW"
            risk_icon = "🟢"
            risk_message = "Low risk of hallucination. Models show good agreement."
        
        return {
            "risk_level": risk_level,
            "risk_icon": risk_icon,
            "risk_score": float(risk_score),
            "risk_message": risk_message,
            "contributing_factors": {
                "entropy": float(entropy),
                "entropy_risk": float(entropy_risk),
                "avg_similarity": float(avg_similarity),
                "similarity_risk": float(similarity_risk),
                "variance": float(variance),
                "variance_risk": float(variance_risk),
                "contradiction_count": contradiction_count,
                "contradiction_risk": float(contradiction_risk),
                "inconsistency_count": inconsistency_count,
                "inconsistency_risk": float(inconsistency_risk),
            },
            "contradictions": contradictions,
            "inconsistencies": inconsistencies,
        }


# Global instance
_estimator = HallucinationRiskEstimator()


async def estimate_hallucination_risk(
    responses: Dict[str, str],
    similarity_matrix: List[List[float]] = None,
    consensus_scores: Dict[str, float] = None
) -> Dict:
    """
    Estimate hallucination risk from model responses.
    
    Args:
        responses: Dict mapping model names to responses
        similarity_matrix: Optional similarity matrix
        consensus_scores: Optional consensus scores
        
    Returns:
        Dict with risk assessment
    """
    return await _estimator.estimate_risk(responses, similarity_matrix, consensus_scores)
