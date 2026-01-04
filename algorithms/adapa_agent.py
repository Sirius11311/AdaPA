"""
AdaPA-Agent: Main Agent Class
Integrates all components for Adaptive Preference Arithmetic personalization
"""

from typing import List, Dict, Tuple, Optional
from openai import OpenAI

from .strength_estimation import StrengthEstimator
from .preference_arithmetic import PreferenceArithmetic
from .prompts import CRS_AGENT_PROMPT


class AdaPAAgent:
    """
    AdaPA-Agent: Adaptive Preference Arithmetic Agent for LLM Personalization.
    
    This agent implements the framework from the paper:
    "Adaptive Preference Arithmetic: Modeling Dynamic Preference Strengths 
    for LLM Agent Personalization"
    
    Key components:
    1. Alignment-Based Strength Estimation: Estimates preference strengths without
       explicit user feedback using dual-side augmentation and LLM-based alignment scoring.
    2. Controllable Personalized Generation: Uses preference arithmetic to combine
       multiple preference-conditioned LLMs with estimated strengths.
    """
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1/",
        estimation_model: str = "gpt-4o-mini",
        generation_model: str = None,
        hf_token: str = None,
        K: int = 3,
        alpha: float = 1.0,
        use_preference_arithmetic: bool = True
    ):
        """
        Initialize the AdaPA-Agent.
        
        Args:
            api_key: API key for OpenAI-compatible endpoint
            api_base: Base URL for the API
            estimation_model: Model name for strength estimation (API-based)
            generation_model: Path to local model for preference arithmetic (optional)
            hf_token: Hugging Face token for model access
            K: Number of interaction augmentations for strength estimation
            alpha: Temperature for softmax normalization
            use_preference_arithmetic: Whether to use preference arithmetic for generation
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.estimation_model = estimation_model
        self.generation_model = generation_model
        self.use_preference_arithmetic = use_preference_arithmetic
        
        # Initialize strength estimator
        self.strength_estimator = StrengthEstimator(
            client=self.client,
            model=estimation_model,
            K=K,
            alpha=alpha
        )
        
        # Initialize preference arithmetic (if generation model provided)
        self.preference_arithmetic = None
        if generation_model and use_preference_arithmetic:
            self.preference_arithmetic = PreferenceArithmetic(
                default_model=generation_model,
                hf_token=hf_token
            )
        
        # Cache for preference chains
        self.preference_chains_cache = {}
    
    def estimate_preference_strengths(
        self,
        preferences: List[str],
        interaction: str,
        use_augmentation: bool = True
    ) -> Tuple[List[float], Dict]:
        """
        Estimate preference strengths based on current interaction.
        
        Args:
            preferences: List of preference descriptions
            interaction: Current user-agent interaction
            use_augmentation: Whether to use dual-side augmentation
            
        Returns:
            Tuple of (list of strengths, debug info dict)
        """
        return self.strength_estimator.estimate_strengths(
            preferences=preferences,
            interaction=interaction,
            augment_preferences=use_augmentation,
            augment_interaction=use_augmentation
        )
    
    def generate_with_preferences(
        self,
        preference_prompts: List[str],
        weights: List[float],
        input_text: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        Generate response using preference arithmetic.
        
        Args:
            preference_prompts: List of preference-conditioned system prompts
            weights: List of weights for each preference
            input_text: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.preference_arithmetic is None:
            raise ValueError("Preference arithmetic not initialized. Provide generation_model.")
        
        # Build formula and generate
        self.preference_arithmetic.build_formula(preference_prompts, weights)
        return self.preference_arithmetic.generate(input_text, max_new_tokens)
    
    def generate_with_api(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0
    ) -> str:
        """
        Generate response using API-based model (fallback when not using preference arithmetic).
        
        Args:
            system_prompt: System prompt combining preferences
            user_message: User message
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.estimation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def recommend(
        self,
        long_term_pref: str,
        short_term_pref: str,
        conversation_history: str,
        candidate_items: List[str],
        max_turns: int,
        current_turn: int,
        rec_num: int = 3,
        w_long: float = None,
        w_short: float = None
    ) -> Tuple[str, Dict]:
        """
        Generate a recommendation based on dynamic preference strengths.
        
        This is the main entry point for conversational recommendation.
        
        Args:
            long_term_pref: Long-term preference description/prompt
            short_term_pref: Short-term preference description/prompt
            conversation_history: Current conversation history
            candidate_items: List of candidate items to recommend from
            max_turns: Maximum conversation turns
            current_turn: Current turn number
            rec_num: Number of items to recommend
            w_long: Pre-computed long-term preference strength (optional)
            w_short: Pre-computed short-term preference strength (optional)
            
        Returns:
            Tuple of (recommendation response, debug info)
        """
        debug_info = {}
        
        # Step 1: Use pre-computed strengths or estimate them
        if w_long is not None and w_short is not None:
            # Use pre-computed strengths from caller
            debug_info["strengths_source"] = "pre-computed"
            debug_info["strengths"] = {"long_term": w_long, "short_term": w_short}
        else:
            # Estimate preference strengths internally
            w_long, w_short, estimation_debug = self.strength_estimator.estimate_two_preferences(
                long_term_pref=long_term_pref,
                short_term_pref=short_term_pref,
                interaction=conversation_history
            )
            debug_info["strengths_source"] = "estimated"
            debug_info["strengths"] = {"long_term": w_long, "short_term": w_short}
            debug_info["estimation_debug"] = estimation_debug
        
        # Step 2: Generate recommendation
        if self.preference_arithmetic is not None and self.use_preference_arithmetic:
            # Use preference arithmetic for generation
            long_term_prompt = f"You are a recommender focused on long-term preferences:\n{long_term_pref}"
            short_term_prompt = f"You are a recommender focused on short-term preferences:\n{short_term_pref}"
            
            self.preference_arithmetic.build_two_preference_formula(
                long_term_prompt=long_term_prompt,
                short_term_prompt=short_term_prompt,
                w_long=w_long,
                w_short=w_short
            )
            
            input_text = f"""Based on the conversation and user preferences, recommend exactly {rec_num} movies.

IMPORTANT: You MUST select movies ONLY from the Candidate Items list below. Do NOT recommend any movies outside this list.

Conversation History:
{conversation_history}

Candidate Items (select ONLY from this list):
{candidate_items}

Your {rec_num} recommendations (must be from Candidate Items):"""
            
            response = self.preference_arithmetic.generate(input_text)
        else:
            # Use API-based generation with combined prompt
            system_prompt = CRS_AGENT_PROMPT.format(
                rec_num=rec_num,
                candidate_items=candidate_items,
                max_turns=max_turns,
                current_turn=current_turn,
                conversation_history=conversation_history
            )
            
            # Add preference strength info to prompt
            preference_info = f"""
Based on the current interaction, the estimated preference strengths are:
- Long-term preference weight: {w_long:.2f}
- Short-term preference weight: {w_short:.2f}

Long-term preference: {long_term_pref}
Short-term preference: {short_term_pref}

Please adjust your recommendations to reflect these preference weights.
"""
            system_prompt = preference_info + "\n" + system_prompt
            
            response = self.generate_with_api(
                system_prompt=system_prompt,
                user_message="Please provide your recommendation based on the conversation."
            )
        
        debug_info["response"] = response
        return response, debug_info
    
    def should_use_preference_arithmetic(
        self,
        weights: List[float],
        threshold: float = 0.1
    ) -> bool:
        """
        Determine if preference arithmetic should be used based on weight difference.
        
        If weights are too similar, preference arithmetic may not provide benefit.
        
        Args:
            weights: List of preference weights
            threshold: Minimum difference required
            
        Returns:
            True if preference arithmetic should be used
        """
        if len(weights) < 2:
            return False
        
        max_w = max(weights)
        min_w = min(weights)
        return (max_w - min_w) >= threshold


class AdaPACRSAgent(AdaPAAgent):
    """
    AdaPA-Agent specialized for Conversational Recommendation Systems.
    
    Extends AdaPAAgent with CRS-specific functionality.
    """
    
    def __init__(
        self,
        name: str,
        rec_num: int,
        config: Dict,
        **kwargs
    ):
        """
        Initialize the AdaPA CRS Agent.
        
        Args:
            name: Agent name
            rec_num: Number of recommendations per turn
            config: Configuration dict with api_key, api_base, model
            **kwargs: Additional arguments for AdaPAAgent
        """
        super().__init__(
            api_key=config.get("api_key"),
            api_base=config.get("api_base", "https://api.openai.com/v1/"),
            estimation_model=config.get("model", "gpt-4o-mini"),
            **kwargs
        )
        
        self.name = name
        self.rec_num = rec_num
    
    def generate_response(
        self,
        candidate_items: List[str],
        conversation_history: str,
        max_turns: int,
        current_turn: int,
        long_term_cot: str = None,
        short_term_cot: str = None,
        w_long: float = None,
        w_short: float = None
    ) -> Tuple[str, Dict]:
        """
        Generate CRS response with adaptive preference handling.
        
        Args:
            candidate_items: List of candidate items
            conversation_history: Formatted conversation history
            max_turns: Maximum conversation turns
            current_turn: Current turn number
            long_term_cot: Long-term preference chain (from augment_preference)
            short_term_cot: Short-term preference chain (from augment_preference)
            w_long: Pre-computed long-term preference strength (optional)
            w_short: Pre-computed short-term preference strength (optional)
            
        Returns:
            Tuple of (response, debug info)
        """
        # Use COT prompts if provided, otherwise use general preference
        long_term = long_term_cot or ""
        short_term = short_term_cot or ""
        
        if long_term and short_term:
            # Use AdaPA with pre-computed or dynamically estimated strengths
            return self.recommend(
                long_term_pref=long_term,
                short_term_pref=short_term,
                conversation_history=conversation_history,
                candidate_items=candidate_items,
                max_turns=max_turns,
                current_turn=current_turn,
                rec_num=self.rec_num,
                w_long=w_long,
                w_short=w_short
            )
        else:
            # Fallback to simple generation
            system_prompt = CRS_AGENT_PROMPT.format(
                rec_num=self.rec_num,
                candidate_items=candidate_items,
                max_turns=max_turns,
                current_turn=current_turn,
                conversation_history=conversation_history
            )
            
            response = self.generate_with_api(
                system_prompt=system_prompt,
                user_message="Please provide your recommendation."
            )
            
            return response, {"mode": "simple"}


if __name__ == "__main__":
    """Test the AdaPA-Agent."""
    # Configuration
    config = {
        "api_key": "your-api-key",
        "api_base": "https://api.openai.com/v1/",
        "model": "gpt-4o-mini"
    }
    
    # Create agent
    agent = AdaPACRSAgent(
        name="CRS",
        rec_num=3,
        config=config,
        K=2
    )
    
    # Test conversation
    conversation = """
    User: I'm looking for a thriller with some dark humor.
    CRS: Do you prefer psychological thrillers or action thrillers?
    User: Psychological ones, with unexpected twists.
    """
    
    candidate_items = ["Fight Club", "The Notebook", "Inception", "Legally Blonde"]
    
    long_term_cot = "User prefers psychological thrillers with dark themes and complex plots."
    short_term_cot = "User is currently in the mood for something with dark humor and twists."
    
    # Generate recommendation
    response, debug = agent.generate_response(
        candidate_items=candidate_items,
        conversation_history=conversation,
        max_turns=5,
        current_turn=2,
        long_term_cot=long_term_cot,
        short_term_cot=short_term_cot
    )
    
    print(f"Response: {response}")
    print(f"\nPreference Strengths: {debug.get('strengths', {})}")

