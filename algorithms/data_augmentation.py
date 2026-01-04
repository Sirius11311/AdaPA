"""
Dual-Side Data Augmentation Module
Implements Preference-side (Prompt 1) and Interaction-side (Prompt 2) augmentation
"""

import json
import re
from typing import List, Dict, Optional
from openai import OpenAI

from .prompts import PREFERENCE_AUGMENTATION_PROMPT, INTERACTION_AUGMENTATION_PROMPT


class DualSideAugmentation:
    """
    Implements dual-side data augmentation for AdaPA-Agent.
    - Preference-side: Generates structured preference chains with semantic granularity
    - Interaction-side: Generates K semantically equivalent but lexically diverse conversations
    """
    
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        """
        Initialize the augmentation module.
        
        Args:
            client: OpenAI client instance
            model: Model name to use for augmentation
        """
        self.client = client
        self.model = model
    
    def augment_preference(self, preference: str, candidate_items: List[str] = None) -> Dict:
        """
        Augment a preference description into a structured preference chain.
        Uses Prompt 1 (Reasoning Augmentation) from the paper.
        
        Args:
            preference: User preference description
            candidate_items: List of candidate items to select examples from (max 3)
            
        Returns:
            Dictionary containing:
            - raw: Original/high-level form of the preference
            - refined: Context-aware, clearer reformulation
            - example: List of representative items from candidates (max 3)
        """
        # Format candidate items as a string
        if candidate_items:
            candidates_str = ", ".join(candidate_items)
        else:
            candidates_str = "No specific candidates provided"
        
        prompt = PREFERENCE_AUGMENTATION_PROMPT.format(
            USER_PREFERENCE_CONTEXT=preference,
            CANDIDATE_ITEMS=candidates_str
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_str = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = result_text
            
            preference_chain = json.loads(json_str)
            
            # Validate structure
            if not all(key in preference_chain for key in ["raw", "refined", "example"]):
                raise ValueError("Missing required keys in preference chain")
                
            return preference_chain
            
        except (json.JSONDecodeError, ValueError):
            # Fallback: create a basic preference chain
            return {
                "raw": preference,
                "refined": preference,
                "example": []
            }
    
    def augment_interaction(self, interaction: str, K: int = 3) -> List[str]:
        """
        Generate K semantically equivalent but lexically diverse conversations.
        Uses Prompt 2 (Intuition Augmentation) from the paper.
        
        Args:
            interaction: Original user-agent interaction
            K: Number of augmented conversations to generate
            
        Returns:
            List of augmented interactions (including original)
        """
        if K <= 0:
            return [interaction]
        
        prompt = INTERACTION_AUGMENTATION_PROMPT.format(
            K=K,
            INTERACTION=interaction
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse the augmented conversations
        augmented_conversations = [interaction]  # Always include original
        
        # Try to split by common separators
        if "---" in result_text:
            parts = result_text.split("---")
        elif "\n\n\n" in result_text:
            parts = result_text.split("\n\n\n")
        elif "Conversation" in result_text:
            # Split by "Conversation 1:", "Conversation 2:", etc.
            parts = re.split(r'Conversation\s*\d+[:\.]?\s*', result_text)
            parts = [p for p in parts if p.strip()]
        else:
            # Treat entire response as one augmented conversation
            parts = [result_text]
        
        for part in parts:
            cleaned = part.strip()
            if cleaned and cleaned != interaction:
                augmented_conversations.append(cleaned)
        
        return augmented_conversations[:K + 1]  # Original + K augmented
    
    def augment_both(self, preference: str, interaction: str, K: int = 3, candidate_items: List[str] = None) -> tuple:
        """
        Perform dual-side augmentation on both preference and interaction.
        
        Args:
            preference: User preference description
            interaction: User-agent interaction
            K: Number of augmented interactions to generate
            candidate_items: List of candidate items to select examples from
            
        Returns:
            Tuple of (preference_chain, augmented_interactions)
        """
        preference_chain = self.augment_preference(preference, candidate_items)
        augmented_interactions = self.augment_interaction(interaction, K)
        
        return preference_chain, augmented_interactions


if __name__ == "__main__":
    """Test the dual-side augmentation module."""
    client = OpenAI(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1/"
    )
    
    augmenter = DualSideAugmentation(client)
    
    # Test preference augmentation
    test_preference = "I enjoy thrillers with elements of gore, adult mature themes, and humor."
    
    print("Testing Preference Augmentation...")
    preference_chain = augmenter.augment_preference(test_preference)
    print(f"Preference Chain: {json.dumps(preference_chain, indent=2)}")
    
    # Test interaction augmentation
    test_interaction = """
    User: I'm looking for a thriller that has some elements of gore and maybe a bit of humor.
    CRS: Have you seen "The Cabin in the Woods"? It's a thriller with a unique twist.
    User: That sounds interesting, but I'm looking for something with more intensity.
    """
    
    print("\nTesting Interaction Augmentation...")
    augmented = augmenter.augment_interaction(test_interaction, K=2)
    for i, conv in enumerate(augmented):
        print(f"\n--- Conversation {i} ---")
        print(conv)

