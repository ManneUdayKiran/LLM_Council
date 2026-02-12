"""Test hallucination detection functionality."""

import asyncio
from hallucination_detector import estimate_hallucination_risk

async def test_low_risk():
    """Test with high agreement - should be LOW risk."""
    print("\n=== TEST 1: High Agreement (LOW RISK) ===")
    responses = {
        "model1": "The capital of France is Paris. It is located in northern France.",
        "model2": "Paris is the capital city of France, situated in the north.",
        "model3": "France's capital is Paris, which is in northern France."
    }
    
    # High similarity matrix (all models agree)
    similarity_matrix = [
        [1.0, 0.95, 0.93],
        [0.95, 1.0, 0.94],
        [0.93, 0.94, 1.0]
    ]
    
    consensus_scores = {
        "model1": 0.94,
        "model2": 0.945,
        "model3": 0.935
    }
    
    risk = await estimate_hallucination_risk(responses, similarity_matrix, consensus_scores)
    print(f"Risk Level: {risk['risk_level']} {risk['risk_icon']}")
    print(f"Risk Score: {risk['risk_score']:.3f}")
    print(f"Message: {risk['risk_message']}")
    print(f"Contradictions: {len(risk.get('contradictions', []))}")
    print(f"Inconsistencies: {len(risk.get('inconsistencies', []))}")

async def test_high_risk():
    """Test with contradictions - should be HIGH risk."""
    print("\n=== TEST 2: Contradictions (HIGH RISK) ===")
    responses = {
        "model1": "The answer is yes, it is safe to do this.",
        "model2": "No, this is not safe and should be avoided.",
        "model3": "The temperature is 100 degrees Fahrenheit."
    }
    
    # Low similarity matrix (models disagree)
    similarity_matrix = [
        [1.0, 0.2, 0.3],
        [0.2, 1.0, 0.25],
        [0.3, 0.25, 1.0]
    ]
    
    consensus_scores = {
        "model1": 0.25,
        "model2": 0.225,
        "model3": 0.275
    }
    
    risk = await estimate_hallucination_risk(responses, similarity_matrix, consensus_scores)
    print(f"Risk Level: {risk['risk_level']} {risk['risk_icon']}")
    print(f"Risk Score: {risk['risk_score']:.3f}")
    print(f"Message: {risk['risk_message']}")
    print(f"Contradictions: {len(risk.get('contradictions', []))}")
    if risk.get('contradictions'):
        for c in risk['contradictions']:
            print(f"  - {c['models'][0]} vs {c['models'][1]}: {c['type']}")

async def test_medium_risk():
    """Test with inconsistencies - should be MEDIUM risk."""
    print("\n=== TEST 3: Inconsistencies (MEDIUM RISK) ===")
    responses = {
        "model1": "The event happened in 2020 with 500 attendees.",
        "model2": "This occurred in 2021 with approximately 750 people.",
        "model3": "The event took place in 2020 with around 600 participants."
    }
    
    # Medium similarity (some agreement)
    similarity_matrix = [
        [1.0, 0.6, 0.7],
        [0.6, 1.0, 0.65],
        [0.7, 0.65, 1.0]
    ]
    
    consensus_scores = {
        "model1": 0.65,
        "model2": 0.625,
        "model3": 0.675
    }
    
    risk = await estimate_hallucination_risk(responses, similarity_matrix, consensus_scores)
    print(f"Risk Level: {risk['risk_level']} {risk['risk_icon']}")
    print(f"Risk Score: {risk['risk_score']:.3f}")
    print(f"Message: {risk['risk_message']}")
    print(f"Inconsistencies: {len(risk.get('inconsistencies', []))}")
    if risk.get('inconsistencies'):
        for i in risk['inconsistencies']:
            print(f"  - {i['models'][0]} vs {i['models'][1]}: {i['type']}")

async def main():
    """Run all tests."""
    await test_low_risk()
    await test_high_risk()
    await test_medium_risk()
    print("\n=== ALL TESTS COMPLETE ===\n")

if __name__ == "__main__":
    asyncio.run(main())
