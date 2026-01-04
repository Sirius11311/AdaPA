"""
Preference Strength Estimation Module
Integrates dual-side augmentation and alignment scoring to compute normalized preference strengths
"""

import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI

from .data_augmentation import DualSideAugmentation
from .alignment_scorer import AlignmentScorer


class StrengthEstimator:
    """
    Estimates preference strengths (ω_i) using alignment-based scoring.
    
    The estimation process:
    1. Augment each preference using Prompt 1 (preference-side augmentation)
    2. Augment the interaction using Prompt 2 (interaction-side augmentation)
    3. Score each augmented interaction against each preference chain using Prompt 3
    4. Normalize scores to get preference strengths
    """
    
    def __init__(
        self, 
        client: OpenAI, 
        model: str = "gpt-4o-mini",
        K: int = 3,
        alpha: float = 1.0
    ):
        """
        Initialize the strength estimator.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
            K: Number of interaction augmentations to generate
            alpha: Temperature parameter for softmax normalization
        """
        self.client = client
        self.model = model
        self.K = K
        self.alpha = alpha
        
        self.augmenter = DualSideAugmentation(client, model)
        self.scorer = AlignmentScorer(client, model)
    
    def estimate_strengths(
        self,
        preferences: List[str],
        interaction: str,
        augment_preferences: bool = True,
        augment_interaction: bool = True,
        candidate_items: List[str] = None
    ) -> Tuple[List[float], Dict]:
        """
        Estimate preference strengths for multiple preferences given an interaction.
        
        Args:
            preferences: List of preference descriptions
            interaction: User-agent interaction text
            augment_preferences: Whether to augment preferences (Prompt 1)
            augment_interaction: Whether to augment interaction (Prompt 2)
            candidate_items: List of candidate items for example selection
            
        Returns:
            Tuple of (normalized strengths, debug info dict)
        """
        debug_info = {
            "preference_chains": [],
            "augmented_interactions": [],
            "raw_scores": [],
            "avg_scores": []
        }
        
        # Step 1: Augment preferences (Prompt 1)
        preference_chains = []
        for pref in preferences:
            if augment_preferences:
                chain = self.augmenter.augment_preference(pref, candidate_items)
            else:
                chain = {"raw": pref, "refined": pref, "example": []}
            preference_chains.append(chain)
        debug_info["preference_chains"] = preference_chains
        
        # Step 2: Augment interaction (Prompt 2)
        if augment_interaction and self.K > 0:
            augmented_interactions = self.augmenter.augment_interaction(interaction, self.K)
        else:
            augmented_interactions = [interaction]
        debug_info["augmented_interactions"] = augmented_interactions
        
        # Step 3: Score each interaction against each preference chain (Prompt 3)
        all_scores = []
        for chain in preference_chains:
            scores = self.scorer.score_multiple(augmented_interactions, chain)
            all_scores.append(scores)
        debug_info["raw_scores"] = all_scores
        
        # Step 4: Compute average scores per preference
        avg_scores = [np.mean(scores) for scores in all_scores]
        debug_info["avg_scores"] = avg_scores
        
        # Step 5: Normalize using softmax
        strengths = self._softmax_normalize(avg_scores)
        
        return strengths, debug_info
    
    def estimate_two_preferences(
        self,
        long_term_pref: str,
        short_term_pref: str,
        interaction: str,
        augment: bool = True,
        candidate_items: List[str] = None
    ) -> Tuple[float, float, Dict]:
        """
        Estimate strengths for long-term and short-term preferences.
        Convenience method for the common two-preference case.
        
        Args:
            long_term_pref: Long-term preference description
            short_term_pref: Short-term preference description
            interaction: User-agent interaction text
            augment: Whether to apply augmentation
            candidate_items: List of candidate items for example selection
            
        Returns:
            Tuple of (long_term_strength, short_term_strength, debug_info)
        """
        preferences = [long_term_pref, short_term_pref]
        strengths, debug_info = self.estimate_strengths(
            preferences, 
            interaction,
            augment_preferences=augment,
            augment_interaction=augment,
            candidate_items=candidate_items
        )
        
        return strengths[0], strengths[1], debug_info
    
    def _softmax_normalize(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using softmax with temperature.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Normalized strengths that sum to 1
        """
        if not scores:
            return []
        
        # Apply temperature scaling
        scores_array = np.array(scores) / self.alpha
        
        # Softmax normalization
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Subtract max for numerical stability
        normalized = exp_scores / np.sum(exp_scores)
        
        return normalized.tolist()
    
    def estimate_with_threshold(
        self,
        preferences: List[str],
        interaction: str,
        threshold: float = 0.1
    ) -> Tuple[List[float], bool]:
        """
        Estimate strengths and determine if preferences are distinguishable.
        
        Args:
            preferences: List of preference descriptions
            interaction: User-agent interaction text
            threshold: Minimum difference required for distinguishable preferences
            
        Returns:
            Tuple of (strengths, is_distinguishable)
        """
        strengths, _ = self.estimate_strengths(preferences, interaction)
        
        if len(strengths) < 2:
            return strengths, False
        
        # Check if the difference between max and min is above threshold
        max_strength = max(strengths)
        min_strength = min(strengths)
        is_distinguishable = (max_strength - min_strength) >= threshold
        
        return strengths, is_distinguishable


if __name__ == "__main__":
    """Test the strength estimation module."""
    client = OpenAI(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1/"
    )
    
    estimator = StrengthEstimator(client, K=2)
    
    # Test preferences
    long_term_pref = "I enjoy psychological thrillers with complex plots and dark themes."
    short_term_pref = "I want something light and fun for a casual movie night."
    
    # Test interaction (aligned more with short-term)
    interaction = """
    User: I'm looking for something fun to watch tonight, nothing too heavy.
    CRS: How about a comedy or an action movie?
    User: Yeah, something entertaining and easy to follow would be great.
    """
    
    print("Testing Strength Estimation...")
    w_long, w_short, debug = estimator.estimate_two_preferences(
        long_term_pref, short_term_pref, interaction
    )
    
    print(f"Long-term preference strength: {w_long:.3f}")
    print(f"Short-term preference strength: {w_short:.3f}")
    print(f"\nDebug info:")
    print(f"Average scores: {debug['avg_scores']}")

