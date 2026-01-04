"""
AdaPA-Agent: Adaptive Preference Arithmetic for LLM Agent Personalization

This module implements the AdaPA-Agent framework from the paper:
"Adaptive Preference Arithmetic: Modeling Dynamic Preference Strengths for LLM Agent Personalization"
"""

from .prompts import (
    PREFERENCE_AUGMENTATION_PROMPT,
    INTERACTION_AUGMENTATION_PROMPT,
    ALIGNMENT_SCORING_PROMPT,
    USER_SIMULATOR_PROMPT,
    CRS_AGENT_PROMPT
)

__all__ = [
    'PREFERENCE_AUGMENTATION_PROMPT',
    'INTERACTION_AUGMENTATION_PROMPT', 
    'ALIGNMENT_SCORING_PROMPT',
    'USER_SIMULATOR_PROMPT',
    'CRS_AGENT_PROMPT',
    'DualSideAugmentation',
    'AlignmentScorer',
    'StrengthEstimator',
    'PreferenceArithmetic',
    'AdaPAAgent',
    'AdaPACRSAgent'
]

# Lazy loading for main classes to avoid import issues
def __getattr__(name):
    if name == 'DualSideAugmentation':
        from .data_augmentation import DualSideAugmentation
        return DualSideAugmentation
    elif name == 'AlignmentScorer':
        from .alignment_scorer import AlignmentScorer
        return AlignmentScorer
    elif name == 'StrengthEstimator':
        from .strength_estimation import StrengthEstimator
        return StrengthEstimator
    elif name == 'PreferenceArithmetic':
        from .preference_arithmetic import PreferenceArithmetic
        return PreferenceArithmetic
    elif name == 'AdaPAAgent':
        from .adapa_agent import AdaPAAgent
        return AdaPAAgent
    elif name == 'AdaPACRSAgent':
        from .adapa_agent import AdaPACRSAgent
        return AdaPACRSAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

