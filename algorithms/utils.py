"""
Utility functions for AdaPA-Agent
"""

from typing import List, Tuple


def conversation_organize(conversation_history: List, advise_flag: bool = False):
    """
    Organize conversation history by separating main conversation from advice.
    
    Args:
        conversation_history: List of (role, message) tuples
        advise_flag: Whether to return advice separately
        
    Returns:
        Organized conversation (and optionally advice list)
    """
    organized_conversation = []
    su_advise_list = []
    for role, message in conversation_history:
        if 'Advise from Simulated User Agent' not in role:
            organized_conversation.append([role, message])
        else:
            su_advise_list.append([role, message])
    
    if advise_flag:
        return [organized_conversation, su_advise_list]
    else:
        return organized_conversation


def conversation_reformat(conversation_history: List[Tuple[str, str]]) -> str:
    """
    Reformat conversation history into a readable string.
    
    Args:
        conversation_history: List of (role, message) tuples
        
    Returns:
        Formatted conversation string
    """
    conversation_stream = ""
    for role, message in conversation_history:
        conversation_stream += f"**{role}**: {message}\n\n"
    return conversation_stream

