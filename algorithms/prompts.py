"""
AdaPA-Agent Prompt Templates
Based on the appendix of the paper "Adaptive Preference Arithmetic: 
Modeling Dynamic Preference Strengths for LLM Agent Personalization"
"""

# Prompt 1: Reasoning Augmentation (Preference-side Augmentation)
PREFERENCE_AUGMENTATION_PROMPT = """Your goal is to help an AI agent better understand user preferences for personalized movie recommendation.

Given the following user preference description:
{USER_PREFERENCE_CONTEXT}

Available candidate movies:
{CANDIDATE_ITEMS}

Your task is to construct a structured preference chain with three levels of semantic granularity.

Please think step by step and provide:
1. raw: the original or high-level form of the preference
2. refined: a context-aware, clearer reformulation of the raw preference
3. example: select UP TO 3 movies from the candidate list above that best match this preference

IMPORTANT: The examples MUST be selected from the candidate movies list above. Do NOT include movies outside this list.

Output the result in the following JSON format:
{{
    "raw": "<fill in raw preference>",
    "refined": "<fill in refined preference>",
    "example": ["<movie_1>", "<movie_2>", "<movie_3>"]
}}
"""

# Prompt 2: Intuition Augmentation (Interaction-side Augmentation)
INTERACTION_AUGMENTATION_PROMPT = """Please generate {K} semantically equivalent but lexically diverse conversations based on the following user-agent interaction:
User-Agent Interaction:
{INTERACTION}

Instructions:
1. Return only the {K} generated interactions, each preserving the number of turns and speaker roles.
2. Keep the user's underlying intent consistent with the original conversation.
3. Use varied wording and sentence structures to enhance linguistic diversity.
"""

# Prompt 3: Measuring Correlation Score (Alignment Scoring)
ALIGNMENT_SCORING_PROMPT = """You are an alignment evaluator tasked with assessing how well a user-agent interaction aligns with a given preference chain.

Given:
- User-Agent Interaction: {INTERACTION}
- Preference Chain: {PREFERENCE_CHAIN}, which includes a raw description, a refined version, and representative examples of a specific user preference.

Instructions:
1. Think step by step.
2. Judge how strongly the interaction reflects the intent or semantics of the given preference.
3. Provide a single alignment score between 0 and 10:
   - 10 = perfectly aligned (preference is clearly implied or reflected)
   - 0 = completely unrelated
4. Only return the numerical score.
"""

# Prompt 4: LLM-Based User Simulator for Conversational Recommendation
USER_SIMULATOR_PROMPT = """You will play the role of a user interacting with a conversational movie recommendation system. Your task is to find a movie that matches your current taste, which is influenced by your preferences.

Role & Behavior Guidelines:
- Engage naturally with the agent by gradually revealing your preferences.
- Focus only on requesting or evaluating movie suggestions based on your preferences.
- Never mention the name of your target movie.

Task Information:
- In this task, your preference is: {prefer_info}.
- In this task, your target movie is: {target_item}.

Simulation Rules:
1. Start with vague intent (e.g., "I want to watch something meaningful").
2. Reveal preference cues as the agent asks follow-up questions.
3. Accept recommendations if they match the target movie.
4. Politely reject unrelated ones and give vague but helpful feedback (e.g., "That's not quite what I'm looking for").
5. Maintain a natural and preference-driven tone throughout.

IMPORTANT: Your role is to simulate a movie enthusiast who is exploring potential movie recommendations, not to reveal the exact title of the target movie—{target_item}. Keep the conversation natural and engaging, and always focus on requesting recommendations or giving feedback based on the suggestions you receive.
"""

# CRS Agent System Prompt for generating recommendations
CRS_AGENT_PROMPT = """You are a **Conversational Recommender System**. Your goal is to engage in dynamic conversations and provide personalized recommendations based on the user's needs.

You have three primary strategies:
1. **Chitchat**: Engage in casual conversation to gather more information about the user's preferences.
2. **Ask**: Directly ask the user for more specific details to refine your understanding of their preferences.
3. **Recommendation**: When you believe you have enough information, recommend **{rec_num}** items that are most likely to be accepted by the user. You must only recommend from the following set of candidate items: **{candidate_items}**.

### Guidelines:
- **Final Turn**: If this is the final turn of the conversation, you **must** make a recommendation.
- **Intermediate Turns**: In earlier turns, you can choose between **Chitchat**, **Ask**, or **Recommendation**.
- **Avoid unnecessary outputs**: Please respond directly as the recommender system.

### Conversation Context:
- **Max Turns**: {max_turns}. This is turn **{current_turn}**.
- **Current Conversation**: {conversation_history}

### Candidate Items: {candidate_items}
"""

