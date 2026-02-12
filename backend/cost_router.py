"""Cost-aware model routing with quality-based escalation."""

from typing import List, Dict, Tuple, Optional
import asyncio

try:
    from .config import MODEL_TIERS, DISAGREEMENT_THRESHOLD
    from .openrouter import query_models_parallel
    from .semantic_consensus import analyze_semantic_consensus
except ImportError:
    from config import MODEL_TIERS, DISAGREEMENT_THRESHOLD
    from openrouter import query_models_parallel
    from semantic_consensus import analyze_semantic_consensus


class CostAwareRouter:
    """Routes queries to models based on cost and quality requirements."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.query_log = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def calculate_cost(self, text: str, tier: str) -> float:
        """Calculate cost for a query."""
        tokens = self.estimate_tokens(text)
        cost_per_1m = MODEL_TIERS[tier]["cost_per_1m_tokens"]
        return (tokens / 1_000_000) * cost_per_1m
    
    async def try_cheap_models_first(
        self, user_query: str
    ) -> Tuple[Dict[str, str], float, str]:
        """
        Try cheap models first and return results with cost.
        
        Returns:
            Tuple of (responses_dict, cost, tier_used)
        """
        tier = "cheap"
        models = MODEL_TIERS[tier]["models"]
        
        messages = [{"role": "user", "content": user_query}]
        responses = await query_models_parallel(models, messages)
        
        # Convert to dict format
        responses_dict = {
            model: resp.get("content", "") 
            for model, resp in responses.items() 
            if resp is not None
        }
        
        # Calculate cost
        total_input_tokens = self.estimate_tokens(user_query) * len(models)
        total_output_tokens = sum(
            self.estimate_tokens(resp) for resp in responses_dict.values()
        )
        cost = ((total_input_tokens + total_output_tokens) / 1_000_000) * MODEL_TIERS[tier]["cost_per_1m_tokens"]
        
        return responses_dict, cost, tier
    
    async def escalate_to_premium(
        self, user_query: str, disagreement_reason: str
    ) -> Tuple[Dict[str, str], float, str]:
        """
        Escalate to premium models when cheap models disagree.
        
        Returns:
            Tuple of (responses_dict, cost, tier_used)
        """
        tier = "premium"
        models = MODEL_TIERS[tier]["models"]
        
        # Add context about why we're escalating
        escalation_context = f"""[ESCALATION NOTICE: This query was escalated to premium models due to: {disagreement_reason}]

{user_query}"""
        
        messages = [{"role": "user", "content": escalation_context}]
        responses = await query_models_parallel(models, messages)
        
        # Convert to dict format
        responses_dict = {
            model: resp.get("content", "") 
            for model, resp in responses.items() 
            if resp is not None
        }
        
        # Calculate cost
        total_input_tokens = self.estimate_tokens(escalation_context) * len(models)
        total_output_tokens = sum(
            self.estimate_tokens(resp) for resp in responses_dict.values()
        )
        cost = ((total_input_tokens + total_output_tokens) / 1_000_000) * MODEL_TIERS[tier]["cost_per_1m_tokens"]
        
        return responses_dict, cost, tier
    
    async def route_with_cost_awareness(
        self, user_query: str
    ) -> Dict:
        """
        Main routing logic: try cheap, escalate if needed.
        
        Returns:
            Dict with responses, consensus, cost breakdown, and routing decisions
        """
        routing_log = []
        total_cost = 0.0
        
        # Step 1: Try cheap models first
        routing_log.append({
            "step": 1,
            "action": "Trying cheap models first",
            "tier": "cheap"
        })
        
        cheap_responses, cheap_cost, cheap_tier = await self.try_cheap_models_first(user_query)
        total_cost += cheap_cost
        
        # Step 2: Analyze consensus
        cheap_consensus = await analyze_semantic_consensus(
            cheap_responses,
            user_query,
            generate_merged=False  # Save cost
        )
        
        confidence = cheap_consensus.get("confidence", 0)
        disagreement_detected = cheap_consensus.get("disagreement_detected", False)
        
        routing_log.append({
            "step": 2,
            "action": "Analyzed consensus",
            "confidence": confidence,
            "disagreement": disagreement_detected,
            "cost": cheap_cost
        })
        
        # Step 3: Decide if escalation is needed
        escalated = False
        premium_responses = None
        premium_consensus = None
        premium_cost = 0.0
        
        if disagreement_detected and confidence < DISAGREEMENT_THRESHOLD:
            routing_log.append({
                "step": 3,
                "action": "Escalating to premium models",
                "reason": f"Low confidence ({confidence:.1%}) and disagreement detected"
            })
            
            premium_responses, premium_cost, premium_tier = await self.escalate_to_premium(
                user_query,
                f"Low confidence ({confidence:.1%}) among cheap models"
            )
            total_cost += premium_cost
            escalated = True
            
            # Analyze premium consensus
            premium_consensus = await analyze_semantic_consensus(
                premium_responses,
                user_query,
                generate_merged=True  # Worth the cost for premium
            )
            
            routing_log.append({
                "step": 4,
                "action": "Premium models completed",
                "confidence": premium_consensus.get("confidence", 0),
                "cost": premium_cost
            })
        
        # Step 4: Compare quality vs cost
        quality_report = self._generate_quality_report(
            cheap_consensus, premium_consensus, cheap_cost, premium_cost, escalated
        )
        
        return {
            "cheap_responses": cheap_responses,
            "cheap_consensus": cheap_consensus,
            "premium_responses": premium_responses,
            "premium_consensus": premium_consensus,
            "escalated": escalated,
            "total_cost": total_cost,
            "cost_breakdown": {
                "cheap_tier": cheap_cost,
                "premium_tier": premium_cost,
            },
            "routing_log": routing_log,
            "quality_report": quality_report,
        }
    
    def _generate_quality_report(
        self, cheap_consensus, premium_consensus, cheap_cost, premium_cost, escalated
    ) -> Dict:
        """Generate quality vs cost comparison."""
        report = {
            "cheap_quality": cheap_consensus.get("confidence", 0),
            "cheap_cost": cheap_cost,
            "escalated": escalated,
        }
        
        if escalated and premium_consensus:
            premium_quality = premium_consensus.get("confidence", 0)
            report["premium_quality"] = premium_quality
            report["premium_cost"] = premium_cost
            
            quality_improvement = premium_quality - report["cheap_quality"]
            cost_increase = premium_cost
            
            report["quality_improvement"] = quality_improvement
            report["cost_increase"] = cost_increase
            
            # Value score (quality improvement per dollar spent)
            if cost_increase > 0:
                report["value_score"] = quality_improvement / cost_increase
            else:
                report["value_score"] = 0
            
            # Recommendation
            if quality_improvement > 0.2 and report["value_score"] > 10:
                report["recommendation"] = "Premium upgrade worth it: significant quality improvement"
            elif quality_improvement < 0.05:
                report["recommendation"] = "Premium upgrade minimal: stick with cheap models"
            else:
                report["recommendation"] = "Premium upgrade moderate: depends on use case"
        else:
            report["recommendation"] = "Cheap models sufficient: no escalation needed"
        
        return report


# Global router instance
_router = CostAwareRouter()


async def route_query_with_cost_awareness(user_query: str) -> Dict:
    """
    Main entry point for cost-aware routing.
    
    Args:
        user_query: User's question
        
    Returns:
        Dict with responses, consensus, costs, and routing decisions
    """
    return await _router.route_with_cost_awareness(user_query)
