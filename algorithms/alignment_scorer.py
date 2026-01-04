"""
Alignment Scorer Module
Implements LLM-based alignment scoring using Prompt 3 from the paper
"""

import re
from typing import Dict, List, Union
from openai import OpenAI

from .prompts import ALIGNMENT_SCORING_PROMPT


class AlignmentScorer:
    """
    LLM-based alignment scorer that evaluates how well a user-agent interaction
    aligns with a given preference chain.
    
    Based on Prompt 3 (Measuring Correlation Score) from the paper.
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        """
        Initialize the alignment scorer.
        
        Args:
            client: OpenAI client instance
            model: Model name to use for scoring
        """
        self.client = client
        self.model = model
    
    def _format_preference_chain(self, preference_chain: Dict) -> str:
        """
        Format a preference chain dictionary into a readable string.
        
        Args:
            preference_chain: Dictionary with raw, refined, and example keys
            
        Returns:
            Formatted string representation
        """
        if isinstance(preference_chain, str):
            return preference_chain
        
        parts = []
        if "raw" in preference_chain:
            parts.append(f"Raw: {preference_chain['raw']}")
        if "refined" in preference_chain:
            parts.append(f"Refined: {preference_chain['refined']}")
        if "example" in preference_chain:
            examples = preference_chain["example"]
            if isinstance(examples, list):
                examples_str = ", ".join(str(e) for e in examples)
            else:
                examples_str = str(examples)
            parts.append(f"Examples: {examples_str}")
        
        return "\n".join(parts)
    
    def score(self, interaction: str, preference_chain: Union[Dict, str]) -> float:
        """
        Calculate the alignment score between an interaction and a preference chain.
        
        Args:
            interaction: User-agent interaction text
            preference_chain: Preference chain (dict or string)
            
        Returns:
            Alignment score between 0 and 10
        """
        formatted_chain = self._format_preference_chain(preference_chain)
        
        prompt = ALIGNMENT_SCORING_PROMPT.format(
            INTERACTION=interaction,
            PREFERENCE_CHAIN=formatted_chain
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse the score
        try:
            # Try to extract a number from the response
            numbers = re.findall(r'\d+\.?\d*', result_text)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-10 range
                return min(max(score, 0), 10)
            else:
                return 5.0  # Default neutral score
        except (ValueError, IndexError):
            return 5.0  # Default neutral score
    
    def score_multiple(self, interactions: List[str], preference_chain: Union[Dict, str]) -> List[float]:
        """
        Calculate alignment scores for multiple interactions against the same preference chain.
        
        Args:
            interactions: List of user-agent interaction texts
            preference_chain: Preference chain (dict or string)
            
        Returns:
            List of alignment scores
        """
        scores = []
        for interaction in interactions:
            score = self.score(interaction, preference_chain)
            scores.append(score)
        return scores
    
    def score_with_multiple_preferences(
        self, 
        interactions: List[str], 
        preference_chains: List[Union[Dict, str]]
    ) -> Dict[str, List[float]]:
        """
        Calculate alignment scores for multiple interactions against multiple preference chains.
        
        Args:
            interactions: List of user-agent interaction texts
            preference_chains: List of preference chains
            
        Returns:
            Dictionary mapping preference index to list of scores
        """
        results = {}
        for i, pref_chain in enumerate(preference_chains):
            pref_key = f"preference_{i}"
            results[pref_key] = self.score_multiple(interactions, pref_chain)
        return results


if __name__ == "__main__":
    """Test the alignment scorer module."""
    client = OpenAI(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1/"
    )
    
    scorer = AlignmentScorer(client)
    
    # Test preference chain
    preference_chain = {
        "raw": "I enjoy thrillers with gore and dark humor",
        "refined": "User prefers psychological thrillers that blend violence with satirical elements",
        "example": ["American Psycho", "Fight Club", "In Bruges"]
    }
    
    # Test interactions
    test_interaction_aligned = """
    User: I'm looking for a movie with dark humor and some gore.
    CRS: Have you seen Fight Club? It has dark comedy elements.
    User: Yes! That's exactly the kind of vibe I want.
    """
    
    test_interaction_unrelated = """
    User: I want to watch a romantic comedy.
    CRS: How about The Notebook?
    User: That sounds perfect for a cozy night.
    """
    
    print("Testing Alignment Scorer...")
    
    score_aligned = scorer.score(test_interaction_aligned, preference_chain)
    print(f"Score for aligned interaction: {score_aligned}")
    
    score_unrelated = scorer.score(test_interaction_unrelated, preference_chain)
    print(f"Score for unrelated interaction: {score_unrelated}")

