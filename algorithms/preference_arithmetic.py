"""
Preference Arithmetic Module
Implements controllable personalized generation using ModelArithmetic and PromptedLLM

Note: This module requires the 'model-arithmetic' package to be installed.
Install via: pip install model-arithmetic
Or from source: https://github.com/eth-sri/language-model-arithmetic
"""

import os
import torch
from typing import List, Dict, Optional

try:
    from huggingface_hub import login
except ImportError:
    login = None

# Import model_arithmetic as external dependency
try:
    from model_arithmetic import ModelArithmetic, PromptedLLM
except ImportError:
    import warnings
    warnings.warn(
        "Could not import ModelArithmetic. Preference arithmetic will not be available. "
        "Please install via: pip install model-arithmetic"
    )
    ModelArithmetic = None
    PromptedLLM = None

# Global cache for loaded models to avoid reloading across ModelArithmetic instances
_GLOBAL_LOADED_MODELS_CACHE = {}


# Define prompt templates as module-level functions
def llama_prompt_template(formula_string, input_string):
    """Prompt template for Llama 3 style models."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{formula_string}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_string}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


def llama2_prompt_template(formula_string, input_string):
    """Prompt template for Llama 2 style models."""
    return f"<s>[INST]<<SYS>>\n{formula_string}\n<</SYS>>\n\n{input_string} [/INST]"


class PreferenceArithmetic:
    """
    Implements Controllable Personalized Generation using Preference Arithmetic.
    
    This module combines multiple preference-conditioned LLMs using weighted
    linear combination of their next-token distributions:
    
    p(y|x) = Σᵢ ωᵢ · p(y|x, prefᵢ)
    
    where ωᵢ are the preference strengths estimated by StrengthEstimator.
    """
    
    def __init__(
        self,
        default_model: str = None,
        prompt_template: callable = None,
        hf_token: str = None,
        device: str = None
    ):
        """
        Initialize the Preference Arithmetic module.
        
        Args:
            default_model: Path to the default model (e.g., Llama-2-7b-chat-hf)
            prompt_template: Function to format prompts (system_prompt, input) -> full_prompt
            hf_token: Hugging Face token for model access
            device: Device to use ('cuda' or 'cpu')
        """
        self.default_model = default_model
        # Use Llama2 template by default (compatible with Llama-2-7b-chat-hf)
        self.prompt_template = prompt_template or llama2_prompt_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if using local model path
        is_local_model = default_model and (
            default_model.startswith('/') or 
            default_model.startswith('./') or
            os.path.exists(default_model)
        )
        
        # Set offline mode for local models to prevent network requests
        if is_local_model:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        elif hf_token and login is not None:
            # Only login to Hugging Face if using remote models
            login(token=hf_token)
        
        # Storage for created models - these are reused across calls
        self.prompted_llms = {}
        self.current_formula = None
        self.model_arithmetic = None
        self._is_initialized = False
        self._cached_loaded_models = {}
    
    def create_preference_llm(self, name: str, preference_prompt: str, reuse_existing: bool = True):
        """
        Create a PromptedLLM for a specific preference.
        
        Args:
            name: Unique name for this preference LLM
            preference_prompt: The preference-conditioned system prompt
            reuse_existing: If True and LLM with this name exists, update its prompt
            
        Returns:
            PromptedLLM instance
        """
        if PromptedLLM is None:
            raise ImportError(
                "PromptedLLM not available. Please install model-arithmetic: "
                "pip install model-arithmetic"
            )
        
        # Reuse existing LLM if available (just update the system prompt)
        if reuse_existing and name in self.prompted_llms:
            self.prompted_llms[name].set_system_prompt(preference_prompt)
            return self.prompted_llms[name]
        
        prompted_llm = PromptedLLM(
            preference_prompt,
            prompt_template=self.prompt_template,
            run_eager=True
        )
        self.prompted_llms[name] = prompted_llm
        return prompted_llm
    
    def build_formula(
        self,
        preference_prompts: List[str],
        weights: List[float],
        names: Optional[List[str]] = None
    ) -> 'ModelArithmetic':
        """
        Build the arithmetic formula combining multiple preference-conditioned LLMs.
        
        Args:
            preference_prompts: List of preference-conditioned system prompts
            weights: List of weights (strengths) for each preference
            names: Optional names for each preference LLM
            
        Returns:
            ModelArithmetic instance
        """
        if len(preference_prompts) != len(weights):
            raise ValueError("Number of prompts must match number of weights")
        
        if names is None:
            names = [f"pref_{i}" for i in range(len(preference_prompts))]
        
        if ModelArithmetic is None:
            raise ImportError(
                "ModelArithmetic not available. Please install model-arithmetic: "
                "pip install model-arithmetic"
            )
        
        # Always create new PromptedLLMs with current prompts
        prompted_llms = []
        for name, prompt in zip(names, preference_prompts):
            llm = self.create_preference_llm(name, prompt, reuse_existing=False)
            prompted_llms.append(llm)
        
        # Build the weighted formula: Σᵢ ωᵢ · LLMᵢ
        formula = sum(w * llm for w, llm in zip(weights, prompted_llms))
        self.current_formula = formula
        
        global _GLOBAL_LOADED_MODELS_CACHE
        
        if not self._is_initialized:
            # First time: Create ModelArithmetic instance (this loads the model)
            self.model_arithmetic = ModelArithmetic(
                formula, 
                default_model=self.default_model
            )
            # Cache the loaded models globally for future use
            _GLOBAL_LOADED_MODELS_CACHE.update(self.model_arithmetic.loaded_models)
            self._is_initialized = True
        else:
            # Subsequent calls: Inject cached models BEFORE creating new ModelArithmetic
            class CachedModelArithmetic(ModelArithmetic):
                """ModelArithmetic that uses cached loaded models."""
                def load_all_models(self, dtype=torch.bfloat16):
                    # Pre-populate with cached models
                    self.loaded_models.update(_GLOBAL_LOADED_MODELS_CACHE)
                    # Call parent to load any new/missing models
                    super().load_all_models(dtype=dtype)
                    # Update global cache with any new models
                    _GLOBAL_LOADED_MODELS_CACHE.update(self.loaded_models)
            
            self.model_arithmetic = CachedModelArithmetic(
                formula, 
                default_model=self.default_model
            )
        
        return self.model_arithmetic
    
    def build_two_preference_formula(
        self,
        long_term_prompt: str,
        short_term_prompt: str,
        w_long: float,
        w_short: float
    ) -> 'ModelArithmetic':
        """
        Convenience method for the common two-preference case.
        
        Args:
            long_term_prompt: Long-term preference system prompt
            short_term_prompt: Short-term preference system prompt
            w_long: Weight for long-term preference
            w_short: Weight for short-term preference
            
        Returns:
            ModelArithmetic instance
        """
        return self.build_formula(
            preference_prompts=[long_term_prompt, short_term_prompt],
            weights=[w_long, w_short],
            names=["long_term", "short_term"]
        )
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        do_speculation: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using the combined preference arithmetic model.
        
        Args:
            input_text: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            do_speculation: Whether to use speculative decoding
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        if self.model_arithmetic is None:
            raise ValueError("Formula not built. Call build_formula() first.")
        
        output = self.model_arithmetic.generate_text(
            input_text,
            max_new_tokens=max_new_tokens,
            do_speculation=do_speculation,
            **kwargs
        )
        
        # Handle output format (may be list or string)
        if isinstance(output, list):
            return output[0].strip() if output else ""
        return output.strip() if output else ""
    
    def get_formula_string(self) -> str:
        """
        Get a string representation of the current formula.
        
        Returns:
            Formula string like "0.7*long_term + 0.3*short_term"
        """
        if not self.prompted_llms:
            return ""
        
        parts = []
        for name, llm in self.prompted_llms.items():
            parts.append(name)
        return " + ".join(parts)
    
    def update_weights(
        self,
        new_weights: Dict[str, float]
    ) -> 'ModelArithmetic':
        """
        Update the weights of existing preference LLMs and rebuild the formula.
        
        Args:
            new_weights: Dictionary mapping preference names to new weights
            
        Returns:
            Updated ModelArithmetic instance
        """
        if not self.prompted_llms:
            raise ValueError("No preference LLMs created yet. Call build_formula() first.")
        
        # Rebuild formula with new weights
        prompts = []
        weights = []
        names = []
        
        for name, llm in self.prompted_llms.items():
            if name in new_weights:
                prompts.append(llm.system_prompt)
                weights.append(new_weights[name])
                names.append(name)
        
        return self.build_formula(prompts, weights, names)


if __name__ == "__main__":
    """Test the preference arithmetic module."""
    pa = PreferenceArithmetic(
        default_model="/path/to/llama-model",
        hf_token="your-hf-token"
    )
    
    # Example preference prompts
    long_term_prompt = """You are a movie recommender. The user has long-term preferences for:
    - Psychological thrillers with complex plots
    - Dark themes and mature content
    - Movies with unexpected twists
    """
    
    short_term_prompt = """You are a movie recommender. The user currently wants:
    - Something light and entertaining
    - Easy to follow
    - Good for a casual movie night
    """
    
    # Build the formula with estimated weights
    ma = pa.build_two_preference_formula(
        long_term_prompt=long_term_prompt,
        short_term_prompt=short_term_prompt,
        w_long=0.3,
        w_short=0.7
    )
    
    # Generate recommendation
    input_text = "Based on the user's preferences, recommend a movie from: [Movie A, Movie B, Movie C]"
    output = pa.generate(input_text)
    print(f"Generated recommendation: {output}")

