"""Advanced prompt normalization and safety layer."""

import re
from typing import Tuple


class PromptNormalizer:
    """Normalizes and sanitizes user prompts before sending to LLMs."""
    
    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(all\s+)?(previous|prior|above)",
        r"system\s*:\s*you\s+are\s+now",
        r"new\s+instructions?:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<s>.*</s>",
        r"###\s*instruction",
        r"override\s+your\s+programming",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
        r"sudo\s+mode",
        r"admin\s+mode",
    ]
    
    def __init__(self):
        """Initialize the prompt normalizer."""
        self.injection_regex = re.compile(
            "|".join(f"({pattern})" for pattern in self.INJECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE
        )
        # Compile spam pattern separately (backreferences don't work well with join)
        self.repetition_regex = re.compile(r"(.)\1{10,}", re.IGNORECASE)
    
    def detect_injection_attempt(self, prompt: str) -> bool:
        """Check if prompt contains injection attempts."""
        return bool(self.injection_regex.search(prompt))
    
    def detect_spam(self, prompt: str) -> bool:
        """Check if prompt looks like spam."""
        # Check excessive repetition
        if self.repetition_regex.search(prompt):
            return True
        
        # Check excessive capitalization (>70% caps in prompts >20 chars)
        if len(prompt) > 20:
            caps_ratio = sum(1 for c in prompt if c.isupper()) / len(prompt)
            if caps_ratio > 0.7:
                return True
        
        return False
    
    def remove_injection_attempts(self, prompt: str) -> str:
        """Remove suspected injection patterns from prompt."""
        cleaned = self.injection_regex.sub("", prompt)
        return cleaned
    
    def normalize_whitespace(self, prompt: str) -> str:
        """Normalize whitespace in prompt."""
        # Replace multiple spaces with single space
        prompt = re.sub(r" +", " ", prompt)
        
        # Replace multiple newlines with double newline (preserve paragraph breaks)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        
        # Strip leading/trailing whitespace
        prompt = prompt.strip()
        
        return prompt
    
    def neutralize_tone(self, prompt: str) -> str:
        """Convert aggressive or manipulative language to neutral."""
        # Remove excessive punctuation
        prompt = re.sub(r"[!?]{2,}", ".", prompt)
        
        # Remove ALL CAPS words (convert to title case)
        def convert_caps(match):
            word = match.group(0)
            if len(word) > 3:  # Only convert words longer than 3 chars
                return word.capitalize()
            return word
        
        prompt = re.sub(r"\b[A-Z]{4,}\b", convert_caps, prompt)
        
        return prompt
    
    def add_constraints(self, prompt: str) -> str:
        """Add helpful constraints to guide LLM responses."""
        constraints = [
            "Please provide accurate, fact-based responses.",
            "Be concise and clear in your explanation.",
            "If uncertain, acknowledge limitations rather than speculating.",
        ]
        
        # Only add constraints if prompt is substantial
        if len(prompt) > 20:
            constraint_text = " ".join(constraints)
            return f"{prompt}\n\n[Guidelines: {constraint_text}]"
        
        return prompt
    
    def normalize(self, prompt: str) -> Tuple[str, dict]:
        """
        Apply full normalization pipeline to prompt.
        
        Returns:
            Tuple of (normalized_prompt, metadata) where metadata contains
            information about what was detected/modified.
        """
        metadata = {
            "original_length": len(prompt),
            "injection_detected": False,
            "spam_detected": False,
            "modifications_applied": []
        }
        
        # Empty or too short
        if not prompt or len(prompt.strip()) < 2:
            return prompt.strip(), metadata
        
        # Detect issues
        metadata["injection_detected"] = self.detect_injection_attempt(prompt)
        metadata["spam_detected"] = self.detect_spam(prompt)
        
        # Apply normalization steps
        normalized = prompt
        
        # 1. Remove injection attempts
        if metadata["injection_detected"]:
            normalized = self.remove_injection_attempts(normalized)
            metadata["modifications_applied"].append("injection_removal")
        
        # 2. Normalize whitespace
        normalized = self.normalize_whitespace(normalized)
        metadata["modifications_applied"].append("whitespace_normalization")
        
        # 3. Neutralize tone
        original_normalized = normalized
        normalized = self.neutralize_tone(normalized)
        if normalized != original_normalized:
            metadata["modifications_applied"].append("tone_neutralization")
        
        # 4. Add constraints
        normalized = self.add_constraints(normalized)
        metadata["modifications_applied"].append("constraints_added")
        
        metadata["final_length"] = len(normalized)
        
        return normalized, metadata


# Global instance
_normalizer = PromptNormalizer()


def normalize_prompt(prompt: str) -> Tuple[str, dict]:
    """
    Normalize a user prompt before sending to LLMs.
    
    Args:
        prompt: Raw user input
        
    Returns:
        Tuple of (normalized_prompt, metadata)
    """
    return _normalizer.normalize(prompt)
