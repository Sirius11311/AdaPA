"""
User Simulator with AdaPA-Agent Integration
Implements the conversational recommendation system using Adaptive Preference Arithmetic

Based on user_simulator_biCOT.py, this version:
1. Uses Prompt 4 from the paper for user simulation (USER_SIMULATOR_PROMPT)
2. Integrates AdaPA-Agent for CRS recommendation generation with dynamic preference strengths
"""

import openai
from typing import List, Dict, Tuple
import random
import json
from tqdm import tqdm
import threading
from queue import Queue
import os
import sys
import time
import argparse
from datetime import datetime
from copy import deepcopy

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import AdaPA modules from algorithms package
from algorithms.adapa_agent import AdaPAAgent, AdaPACRSAgent
from algorithms.prompts import USER_SIMULATOR_PROMPT
from algorithms.data_augmentation import DualSideAugmentation
from algorithms.strength_estimation import StrengthEstimator
from algorithms.utils import conversation_organize, conversation_reformat


class UserSimulatorAdaPA:
    """
    User Simulator using Prompt 4 from the AdaPA paper.
    
    This implements the LLM-Based User Simulator for Conversational Recommendation
    as described in the paper appendix (Prompt 4).
    
    Key features:
    - Engages naturally with the agent by gradually revealing preferences
    - Never mentions the target movie directly
    - Accepts recommendations matching the target, rejects others with vague feedback
    """
    
    # Prompt 4 from the paper appendix
    SYSTEM_PROMPT = """You will play the role of a user interacting with a conversational movie recommendation system. Your task is to find a movie that matches your current taste, which is influenced by your preferences.

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

IMPORTANT: Your role is to simulate a movie enthusiast who is exploring potential movie recommendations, not to reveal the exact title of the target movie—{target_item}. Keep the conversation natural and engaging, and always focus on requesting recommendations or giving feedback based on the suggestions you receive."""
    
    def __init__(self, config_list: Dict, candidate_size: int, seed: int = None, samples: List[Dict] = None):
        """
        Initialize the user simulator.
        
        Args:
            config_list: Configuration dictionary with 'user' and 'crs' keys
            candidate_size: Number of candidate samples to construct movie pool
            seed: Random seed for reproducibility
            samples: Pre-loaded samples (optional, avoids redundant file loading)
        """
        config_list_user = config_list["user"]
        self.config = config_list_user[0]
        self.client = openai.OpenAI(
            api_key=self.config["api_key"], 
            base_url=self.config["api_base"]
        )
        self.model = self.config["model"]
        
        # Use Prompt 4 from the paper
        self.system_message = self.SYSTEM_PROMPT
      
        self.satisfaction = 0
        self.preference = ""
        self.target_item = ""
        self.user_index = -1
        self.random_state = random.Random(seed)
        self.current_sample_index = 0
        self.candidate_size = candidate_size
        self.candidate_items = []
        self.preference_type = ''
        self.short_term_preference = ''
        self.long_term_preference = ''
        
        # Use pre-loaded samples if provided, otherwise load from file
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_user_data()

    @staticmethod
    def load_all_samples() -> List[Dict]:
        """Load all user data from JSONL file (static method for batch loading)."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data/processed/processed_data_with_recommendations_cleaned.jsonl'
        )
        with open(data_path, 'r') as file:
            return [json.loads(line) for line in file]

    def _load_user_data(self) -> List[Dict]:
        """Load user data from JSONL file and sample."""
        all_data = self.load_all_samples()
        return self.random_state.sample(all_data, self.candidate_size)

    def get_sample(self):
        """Get current sample and initialize candidate items."""
        sample = self.samples[self.current_sample_index]
        self.user_index = sample['index']
        
        # Long-term preference movies (from current sample)
        long_term_movies = deepcopy(sample['movie_name'])
        self.long_term_preference = sample['user_preference']
        
        # Add movies from other samples to create distractor pool
        other_samples = [s for i, s in enumerate(self.samples) if i != self.current_sample_index]
        
        # Use another user's preference as short-term preference
        random_other_sample = self.random_state.choice(other_samples)
        self.short_term_preference = random_other_sample['user_preference']
        
        # Short-term preference movies (from the random other sample)
        short_term_movies = deepcopy(random_other_sample['movie_name'])
        
        # Build candidate items: long-term + short-term + distractors
        self.candidate_items = long_term_movies.copy()
        self.candidate_items.extend(short_term_movies)
        
        # Add other movies as distractors
        for other_sample in other_samples:
            if other_sample != random_other_sample:
                self.candidate_items.extend(other_sample['movie_name'])

        # Remove duplicates and shuffle
        self.candidate_items = list(set(self.candidate_items))
        self.random_state.shuffle(self.candidate_items)
        
        # Target item is sampled from long-term + short-term movies only
        target_pool = list(set(long_term_movies + short_term_movies))
        self.target_item = self.random_state.choice(target_pool)

        # Determine preference type based on which pool the target comes from
        if self.target_item in long_term_movies:
            self.preference_type = 'long-term'
            self.preference = self.long_term_preference
        else:
            self.preference_type = 'short-term'
            self.preference = self.short_term_preference

        self.current_sample_index += 1

    def _get_formatted_system_message(self) -> str:
        """Get the system message formatted with user's preference and target."""
        return self.system_message.format(
            prefer_info=self.preference,
            target_item=self.target_item
        )

    def generate_query(self, conversation_history: List[Tuple[str, str]] = None) -> str:
        """
        Generate user query based on conversation history.
        
        Follows Prompt 4 guidelines:
        - Starts with vague intent
        - Reveals preferences gradually
        - Never mentions target movie
        
        Args:
            conversation_history: List of (speaker, message) tuples
            
        Returns:
            Generated user response
        """
        formatted_system_message = self._get_formatted_system_message()
        
        if conversation_history is None or conversation_history == []:
            # Initial query - start with vague intent
            prompt = """Start a conversation with the movie recommendation system. 
Express a vague intent about wanting to watch something, based on your preferences.
Remember: Do NOT mention your target movie directly. Start with something like "I want to watch something meaningful" or "I'm in the mood for a certain type of movie"."""
        else:
            # Continue conversation - respond based on history
            formatted_history = conversation_reformat(conversation_history)
            prompt = f"""Following is the conversation history between you (the user) and the movie recommendation system:

{formatted_history}

Now, continue the conversation by responding to the recommendation system.
- If the system recommended your target movie, accept it enthusiastically.
- If not, politely reject and provide more hints about your preferences (without mentioning your target movie).
- Keep your response natural and focused on preferences."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formatted_system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=123
        )
        
        content = response.choices[0].message.content
        
        # Safety check: mask target item if accidentally mentioned
        if self.target_item.lower() in content.lower():
            content = content.replace(self.target_item, "[MOVIE]")
            content = content.replace(self.target_item.lower(), "[movie]")
        
        return content


class AdaPACRSAgent_Wrapper:
    """
    Wrapper for AdaPA CRS Agent that generates recommendations
    using adaptive preference arithmetic.
    
    Uses the AdaPA framework:
    1. DualSideAugmentation for preference-side and interaction-side augmentation
    2. StrengthEstimator for computing preference strengths
    3. PreferenceArithmetic for controllable generation
    """
    
    def __init__(
        self, 
        name: str, 
        rec_num: int, 
        config: Dict,
        K: int = 3,
        alpha: float = 1.0,
        generation_model: str = None,
        hf_token: str = None
    ):
        """
        Initialize the AdaPA CRS Agent wrapper.
        
        Args:
            name: Agent name
            rec_num: Number of recommendations per turn
            config: API configuration
            K: Number of interaction augmentations for strength estimation
            alpha: Temperature for softmax normalization
            generation_model: Path to local model for preference arithmetic
            hf_token: Hugging Face token for model access
        """
        self.name = name
        self.rec_num = rec_num
        self.config = config
        self.K = K
        self.alpha = alpha
        self.generation_model = generation_model
        self.hf_token = hf_token
        
        # Initialize OpenAI client for augmentation
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("api_base", "https://api.openai.com/v1/")
        )
        self.model = config.get("model", "gpt-4o-mini")
        
        # Initialize StrengthEstimator for preference strength computation
        self.strength_estimator = StrengthEstimator(
            client=self.client,
            model=self.model,
            K=K,
            alpha=alpha
        )
        
        # Initialize AdaPA CRS Agent with generation model
        self.adapa_agent = AdaPACRSAgent(
            name=name,
            rec_num=rec_num,
            config=config,
            K=K,
            alpha=alpha,
            generation_model=generation_model,
            hf_token=hf_token
        )
    
    def generate_response(
        self,
        candidate_items: List[str],
        conversation_history: List[Tuple[str, str]],
        max_turns: int,
        current_turn: int,
        conversation_trace: List = None,
        user_index: int = None,
        long_term_preference: str = None,
        short_term_preference: str = None
    ) -> Tuple[str, Dict]:
        """
        Generate CRS response using AdaPA preference arithmetic.
        
        Following the AdaPA framework:
        1. Use Prompt 1 (augment_preference) to get preference chains
        2. Use Prompt 2 (augment_interaction) to get augmented interactions
        3. Compute preference strengths using alignment scoring
        4. Generate response with preference arithmetic
        
        Args:
            candidate_items: List of candidate movies
            conversation_history: List of (speaker, message) tuples
            max_turns: Maximum conversation turns
            current_turn: Current turn number
            conversation_trace: Historical conversation traces
            user_index: User index for trace lookup
            long_term_preference: User preference description (long-term)
            short_term_preference: Short-term preference description
            
        Returns:
            Tuple of (response text, debug info)
        """
        debug_info = {}
        
        # Format conversation history
        formatted_conversation = conversation_reformat(conversation_history)
        
        # Extract Preferences
        long_term_pref = long_term_preference or "General movie preferences"
        short_term_pref = short_term_preference or "General movie preferences"
        
        debug_info["raw_preferences"] = {
            "long_term": long_term_pref,
            "short_term": short_term_pref
        }
        
        # Estimate Preference Strengths (Prompt 2 & 3)
        w_long, w_short, estimation_debug = self.strength_estimator.estimate_two_preferences(
            long_term_pref=long_term_pref,
            short_term_pref=short_term_pref,
            interaction=formatted_conversation,
            augment=True,
            candidate_items=candidate_items
        )
        
        # Extract preference chains from estimation debug
        preference_chains = estimation_debug.get("preference_chains", [])
        long_term_chain = preference_chains[0] if len(preference_chains) > 0 else {"raw": long_term_pref}
        short_term_chain = preference_chains[1] if len(preference_chains) > 1 else {"raw": short_term_pref}
        
        debug_info["strengths"] = {"long_term": w_long, "short_term": w_short}
        debug_info["long_term_chain"] = long_term_chain
        debug_info["short_term_chain"] = short_term_chain
        debug_info["estimation_debug"] = estimation_debug
        
        # Generate Response with AdaPA-Agent (Preference Arithmetic)
        response, agent_debug = self.adapa_agent.generate_response(
            candidate_items=candidate_items,
            conversation_history=formatted_conversation,
            max_turns=max_turns,
            current_turn=current_turn,
            long_term_cot=self._chain_to_string(long_term_chain),
            short_term_cot=self._chain_to_string(short_term_chain),
            w_long=w_long,
            w_short=w_short
        )
        
        debug_info.update(agent_debug)
        
        return response, debug_info
    
    def _chain_to_string(self, preference_chain: Dict) -> str:
        """
        Convert a preference chain dict to a string representation.
        """
        parts = []
        if "raw" in preference_chain:
            parts.append(f"Preference: {preference_chain['raw']}")
        if "refined" in preference_chain:
            parts.append(f"Refined: {preference_chain['refined']}")
        if "example" in preference_chain and preference_chain["example"]:
            examples = ", ".join(str(e) for e in preference_chain["example"][:5])
            parts.append(f"Examples: {examples}")
        
        return "\n".join(parts) if parts else str(preference_chain)


class CRSEvaluatorAdaPA:
    """
    Evaluator for CRS with AdaPA-Agent integration.
    
    Orchestrates the conversation between:
    - UserSimulatorAdaPA: Simulates user using Prompt 4
    - AdaPACRSAgent_Wrapper: Generates recommendations with preference arithmetic
    """
    
    def __init__(
        self, 
        user_simulator: UserSimulatorAdaPA, 
        crs_agent: AdaPACRSAgent_Wrapper,
        sample_index: int, 
        log_dir: str, 
        conversation_trace: List[str]
    ):
        self.user_simulator = user_simulator
        self.crs_agent = crs_agent
        self.conversation_history = []
        self.debug_info_list = []
        self.metrics = {
            "turns": 0,
            "user_satisfaction": 0,
            "successful_recommendations": 0,
        }
        self.sample_index = sample_index
        self.log_file = os.path.join(log_dir, f"sample_{self.sample_index}.txt")
        self.conversation_trace = conversation_trace

    def run_conversation(self, max_turns: int = 5):
        """
        Run a complete conversation between user simulator and AdaPA CRS agent.
        """
        self.conversation_history = []
        self.debug_info_list = []
        
        # Generate initial user query using Prompt 4
        user_query = self.user_simulator.generate_query(self.conversation_history)
        self.conversation_history.append(("User", user_query))
        
        for turn in range(max_turns):
            # Generate CRS response using AdaPA-Agent
            crs_response, debug_info = self.crs_agent.generate_response(
                candidate_items=self.user_simulator.candidate_items,
                conversation_history=self.conversation_history,
                max_turns=max_turns,
                current_turn=turn + 1,
                conversation_trace=self.conversation_trace,
                user_index=self.user_simulator.user_index,
                long_term_preference=self.user_simulator.long_term_preference,
                short_term_preference=self.user_simulator.short_term_preference
            )
            
            self.conversation_history.append(("CRS", crs_response))
            self.debug_info_list.append(debug_info)

            # Check if target item is recommended (success!)
            if self.user_simulator.target_item.lower() in crs_response.lower():
                self.metrics["successful_recommendations"] += 1
                self.metrics["turns"] = turn + 1
                break

            # If not last turn, generate user response
            if turn < max_turns - 1:
                user_response = self.user_simulator.generate_query(self.conversation_history)
                self.conversation_history.append(("User", user_response))
            
            self.metrics["turns"] = turn + 1

        self.metrics["user_satisfaction"] = self.user_simulator.satisfaction
        self.write_log()
    
    def write_log(self):
        """Write detailed conversation log to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AdaPA-Agent CRS Evaluation Log\n")
            f.write("=" * 60 + "\n\n")
            
            # User preference information
            f.write("-" * 40 + "\n")
            f.write("User Preference Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Long-term Preference: {self.user_simulator.long_term_preference}\n\n")
            f.write(f"Short-term Preference: {self.user_simulator.short_term_preference}\n\n")
            
            # Target item information
            f.write("-" * 40 + "\n")
            f.write("Target Item Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Target Item: {self.user_simulator.target_item}\n")
            f.write(f"Preference Type: {self.user_simulator.preference_type}\n\n")
            
            # Candidate items
            f.write(f"Candidate Items ({len(self.user_simulator.candidate_items)}): ")
            f.write(", ".join(self.user_simulator.candidate_items))
            f.write("\n\n")
            
            # Write AdaPA preference strength info
            f.write("-" * 40 + "\n")
            f.write("AdaPA Preference Strengths & Chains per Turn:\n")
            f.write("-" * 40 + "\n")
            for i, debug in enumerate(self.debug_info_list):
                f.write(f"\n[Turn {i+1}]\n")
                
                # Write preference strengths
                strengths = debug.get('strengths', {})
                f.write(f"Strengths: Long-term={strengths.get('long_term', 'N/A'):.3f}, ")
                f.write(f"Short-term={strengths.get('short_term', 'N/A'):.3f}\n")
                
                # Write long-term preference chain
                long_term_chain = debug.get('long_term_chain', {})
                f.write(f"Long-term Chain:\n")
                f.write(f"  - Raw: {long_term_chain.get('raw', 'N/A')}\n")
                f.write(f"  - Refined: {long_term_chain.get('refined', 'N/A')}\n")
                f.write(f"  - Examples: {long_term_chain.get('example', [])}\n")
                
                # Write short-term preference chain
                short_term_chain = debug.get('short_term_chain', {})
                f.write(f"Short-term Chain:\n")
                f.write(f"  - Raw: {short_term_chain.get('raw', 'N/A')}\n")
                f.write(f"  - Refined: {short_term_chain.get('refined', 'N/A')}\n")
                f.write(f"  - Examples: {short_term_chain.get('example', [])}\n")
            f.write("\n")
            
            # Write conversation history
            f.write("-" * 40 + "\n")
            f.write("Conversation History:\n")
            f.write("-" * 40 + "\n")
            for speaker, message in self.conversation_history:
                f.write(f"\n**{speaker}**:\n{message}\n")
            
            # Write metrics
            f.write("\n" + "-" * 40 + "\n")
            f.write("Metrics:\n")
            f.write("-" * 40 + "\n")
            for metric, value in self.metrics.items():
                f.write(f"{metric}: {value}\n")
            
            # Write success status
            f.write("\n")
            if self.metrics["successful_recommendations"] > 0:
                f.write("SUCCESS: Target movie was recommended!\n")
            else:
                f.write("FAILED: Target movie was not recommended.\n")


def worker_adapa(
    config_list: Dict, 
    candidate_size: int, 
    rec_num: int, 
    max_turns: int, 
    sample_queue: Queue, 
    result_queue: Queue, 
    log_dir: str, 
    conversation_trace: List,
    seed: int, 
    K: int, 
    alpha: float,
    generation_model: str = None,
    hf_token: str = None
):
    """Worker function for parallel evaluation with AdaPA-Agent."""
    # Create AdaPA CRS Agent ONCE per worker (model only loads once)
    crs_config = config_list['crs'][0]
    crs_agent = AdaPACRSAgent_Wrapper(
        name="CRS",
        rec_num=rec_num,
        config=crs_config,
        K=K,
        alpha=alpha,
        generation_model=generation_model,
        hf_token=hf_token
    )
    
    while True:
        sample_data = sample_queue.get()
        if sample_data is None:
            break
        
        samples, sample_index = sample_data
        
        # Create user simulator with Prompt 4
        user_simulator = UserSimulatorAdaPA(
            config_list, 
            candidate_size, 
            seed=seed, 
            samples=samples
        )
        user_simulator.get_sample()
        
        # Create evaluator (reuses the same crs_agent)
        evaluator = CRSEvaluatorAdaPA(
            user_simulator=user_simulator,
            crs_agent=crs_agent,
            sample_index=sample_index,
            log_dir=log_dir,
            conversation_trace=conversation_trace
        )
        
        evaluator.run_conversation(max_turns)
        result_queue.put(evaluator.metrics)
        
        sample_queue.task_done()


def run_batch_evaluation_adapa(
    config_list: Dict, 
    candidate_size: int, 
    sample_size: int, 
    rec_num: int, 
    max_turns: int = 5, 
    max_workers: int = 5, 
    seed: int = 1, 
    log_dir: str = '', 
    conversation_trace: List[str] = [],
    K: int = 3,
    alpha: float = 1.0,
    generation_model: str = None,
    hf_token: str = None
) -> Dict:
    """
    Run batch evaluation with AdaPA-Agent.
    """
    # Create seed-specific log directory
    seed_log_dir = os.path.join(log_dir, f"seed_{seed}")
    os.makedirs(seed_log_dir, exist_ok=True)

    # Create queues for parallel processing
    sample_queue = Queue()
    result_queue = Queue()

    # Load all samples once using static method
    all_samples = UserSimulatorAdaPA.load_all_samples()
    random.seed(seed)

    # Create sample batches
    for i in range(sample_size):
        samples = random.sample(all_samples, candidate_size)
        sample_queue.put((samples, i))

    # Create and start worker threads
    threads = []
    for _ in range(min(max_workers, sample_size)):
        thread = threading.Thread(
            target=worker_adapa, 
            args=(config_list, candidate_size, rec_num, max_turns, 
                  sample_queue, result_queue, seed_log_dir, conversation_trace,
                  seed, K, alpha, generation_model, hf_token)
        )
        thread.start()
        threads.append(thread)

    # Wait for completion with progress bar
    with tqdm(total=sample_size, desc=f"Processing (Seed {seed})") as pbar:
        completed = 0
        while completed < sample_size:
            new_completed = sample_size - sample_queue.qsize()
            if new_completed > completed:
                pbar.update(new_completed - completed)
                completed = new_completed
            time.sleep(0.1)

    # Stop workers
    for _ in range(min(max_workers, sample_size)):
        sample_queue.put(None)
    for thread in threads:
        thread.join()

    # Collect results
    total_turns = 0
    total_satisfaction = 0
    total_successful_recommendations = 0
    results_count = 0

    while not result_queue.empty():
        metrics = result_queue.get()
        total_turns += metrics["turns"]
        total_satisfaction += metrics["user_satisfaction"]
        total_successful_recommendations += metrics["successful_recommendations"]
        results_count += 1

    # Calculate overall metrics
    avg_turns = total_turns / results_count if results_count > 0 else 0
    avg_satisfaction = total_satisfaction / results_count if results_count > 0 else 0
    success_rate = total_successful_recommendations / results_count if results_count > 0 else 0

    print(f"\nOverall Metrics (Seed {seed}):")
    print(f"  Average Turns: {avg_turns:.2f}")
    print(f"  Success Rate: {success_rate:.2%}")

    return {
        "seed": seed,
        "avg_turns": avg_turns,
        "avg_satisfaction": avg_satisfaction,
        "success_rate": success_rate
    }


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='AdaPA-Agent Conversational Recommender System Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--user_model_id', type=int, default=2, choices=[1,2,3,4,5,6,7,8],
                        help='ID of model to use for user simulation')
    parser.add_argument('--crs_model_id', type=int, default=2, choices=[1,2,3,4,5,6,7,8],
                        help='ID of model to use for CRS')
    parser.add_argument('--api_key', type=str, 
                        default="your-api-key",
                        help='API key for model access')
    parser.add_argument('--api_base', type=str,
                        default="https://api.openai.com/v1/",
                        help='Base URL for API')

    # Experiment parameters  
    parser.add_argument('--seeds', nargs='+', type=int, 
                        default=[1, 10, 100],
                        help='Random seeds to use')
    parser.add_argument('--turns', nargs='+', type=int, default=[3, 5, 7],
                        help='Number of conversation turns')
    parser.add_argument('--sample_size', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--candidate_size', type=int, default=5,
                        help='Number of candidate samples for movie pool')
    parser.add_argument('--rec_num', type=int, default=3,
                        help='Number of recommendations per turn')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of parallel workers')
    parser.add_argument('--conversation_trace_path', type=str,
                        default='../data/processed/filtered_test_data_K_5_M_5.jsonl',
                        help='Path to conversation trace data')
    
    # AdaPA-specific parameters
    parser.add_argument('--K', type=int, default=3,
                        help='Number of interaction augmentations for strength estimation')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Temperature for softmax normalization')
    
    # Preference Arithmetic model parameters
    parser.add_argument('--generation_model', type=str, 
                        default=None,
                        help='Path to local model for preference arithmetic generation')
    parser.add_argument('--hf_token', type=str,
                        default=None,
                        help='Hugging Face token for model access')
    parser.add_argument('--use_preference_arithmetic', action='store_true',
                        help='Enable preference arithmetic with local model (requires GPU)')

    args = parser.parse_args()

    # Model dictionary
    model_dict = {
        1: 'glm-4-flash',
        2: 'gpt-4o-mini', 
        3: 'ERNIE-Lite-8K',
        4: 'gpt-4o-mini-2024-07-18',
        5: 'deepseek-chat',
        6: 'llama2-13b-chat-v2',
        7: 'llama2-7b-chat-v2',
        8: 'Doubao-pro-128k'
    }

    # Build config
    config_list = {
        "user": [{
            "model": model_dict[args.user_model_id],
            "api_key": args.api_key,
            "api_base": args.api_base,
        }],
        "crs": [{
            "model": model_dict[args.crs_model_id], 
            "api_key": args.api_key,
            "api_base": args.api_base,
        }]
    }

    # Load conversation trace
    conversation_trace = []
    trace_path = os.path.join(current_dir, args.conversation_trace_path)
    if os.path.exists(trace_path):
        with open(trace_path, 'r') as file:
            conversation_trace = [json.loads(line).get("history_data", []) for line in file]
    else:
        print(f"Warning: Conversation trace file not found at {trace_path}")

    # Determine generation model
    generation_model = args.generation_model if args.use_preference_arithmetic else None
    hf_token = args.hf_token if args.use_preference_arithmetic else None

    print("=" * 60)
    print("AdaPA-Agent CRS Evaluation")
    print("=" * 60)
    print(f"User Model: {model_dict[args.user_model_id]}")
    print(f"CRS Model: {model_dict[args.crs_model_id]}")
    print(f"Sample Size: {args.sample_size}")
    print(f"K (Augmentation): {args.K}")
    print(f"Alpha: {args.alpha}")
    if args.use_preference_arithmetic:
        print(f"Preference Arithmetic: ENABLED")
        print(f"Generation Model: {args.generation_model}")
    else:
        print(f"Preference Arithmetic: DISABLED (using API-based generation)")
    print("=" * 60)

    for max_turns in args.turns:
        # Create log directory
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs',
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(log_dir, exist_ok=True)

        all_results = []
        print(f"\n>>> Evaluating with max_turns={max_turns}")
        
        for seed in args.seeds:
            random.seed(seed)
            result = run_batch_evaluation_adapa(
                config_list=config_list,
                candidate_size=args.candidate_size,
                sample_size=args.sample_size,
                rec_num=args.rec_num,
                max_turns=max_turns,
                max_workers=args.max_workers,
                seed=seed,
                log_dir=log_dir,
                conversation_trace=conversation_trace,
                K=args.K,
                alpha=args.alpha,
                generation_model=generation_model,
                hf_token=hf_token
            )
            all_results.append(result)

        # Calculate average metrics
        avg_turns = sum(r["avg_turns"] for r in all_results) / len(all_results)
        avg_satisfaction = sum(r["avg_satisfaction"] for r in all_results) / len(all_results)
        avg_success_rate = sum(r["success_rate"] for r in all_results) / len(all_results)

        # Write summary
        summary_path = os.path.join(log_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("AdaPA-Agent CRS Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write("Configuration:\n")
            f.write(f"  User Model: {model_dict[args.user_model_id]}\n")
            f.write(f"  CRS Model: {model_dict[args.crs_model_id]}\n")
            f.write(f"  Sample Size: {args.sample_size}\n")
            f.write(f"  Candidate Size: {args.candidate_size}\n")
            f.write(f"  Recommendations per Turn: {args.rec_num}\n")
            f.write(f"  Max Turns: {max_turns}\n")
            f.write(f"  K (Interaction Augmentation): {args.K}\n")
            f.write(f"  Alpha (Softmax Temperature): {args.alpha}\n")
            f.write(f"  Preference Arithmetic: {'ENABLED' if args.use_preference_arithmetic else 'DISABLED'}\n")
            if args.use_preference_arithmetic:
                f.write(f"  Generation Model: {args.generation_model}\n")
            f.write("\n")

            f.write("Results per Seed:\n")
            for result in all_results:
                f.write(f"\n  Seed {result['seed']}:\n")
                f.write(f"    Average Turns: {result['avg_turns']:.2f}\n")
                f.write(f"    Success Rate: {result['success_rate']:.2%}\n")

            f.write(f"\nOverall Average (across {len(all_results)} seeds):\n")
            f.write(f"  Average Turns: {avg_turns:.2f}\n")
            f.write(f"  Average Success Rate: {avg_success_rate:.2%}\n")

        print(f"\n>>> Results for max_turns={max_turns}:")
        print(f"    Average Success Rate: {avg_success_rate:.2%}")
        print(f"    Results saved to: {log_dir}")

