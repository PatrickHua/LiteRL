"""
Math reward computation utilities.
Matches PipelineRL's reward computation logic.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def verify_math_answer(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    """
    Verify if a math answer is correct.
    Matches PipelineRL's verify_math function logic.
    
    Args:
        prediction: Generated text from model
        gold: Ground truth answer (should be in \\boxed{} format)
        strict: Whether to use strict matching
        max_prediction_length: Maximum length for boxed prediction
        
    Returns:
        One of: "correct", "wrong", "no_answer", "unparsable"
    """
    try:
        # Input validation
        if not isinstance(prediction, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")
        
        # Look for \\boxed{} in prediction (from the right, like PipelineRL)
        boxed_start = prediction.rfind("\\boxed{")
        
        if boxed_start < 0:
            return "no_answer"
        
        boxed_prediction = prediction[boxed_start:]
        
        # Check for empty boxed
        if "\\boxed{}" in boxed_prediction:
            return "no_answer"
        
        # Check length
        if len(boxed_prediction) > max_prediction_length:
            return "unparsable"
        
        # Extract content inside \\boxed{}
        match = re.search(r"\\boxed\{([^}]+)\}", boxed_prediction)
        if not match:
            return "unparsable"
        
        pred_answer = match.group(1).strip()
        
        # Extract gold answer (should already be in \\boxed{} format)
        gold_match = re.search(r"\\boxed\{([^}]+)\}", gold)
        if not gold_match:
            # If gold doesn't have \\boxed{}, try to extract it directly
            gold_answer = gold.strip()
        else:
            gold_answer = gold_match.group(1).strip()
        
        # Simple string matching for now
        # TODO: Use math_verify library for symbolic math verification like PipelineRL
        # For now, we do normalized comparison
        pred_normalized = re.sub(r'\s+', ' ', pred_answer)
        gold_normalized = re.sub(r'\s+', ' ', gold_answer)
        
        if pred_normalized == gold_normalized:
            return "correct"
        else:
            return "wrong"
                
    except Exception as e:
        logger.warning(f"Error verifying answer: {e}")
        return "unparsable"


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty.
    Matches PipelineRL's length_penalty function.
    
    Args:
        max_length: Maximum sequence length
        sequence_length: Actual sequence length
        buffer_tokens: Buffer tokens before penalty kicks in
        
    Returns:
        Penalty value (negative or zero)
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.0


def compute_math_reward(
    rollout: Dict[str, Any],
    problem: Dict[str, Any],
    reward_config: Optional[Dict[str, Any]] = None,
    discount_factor: float = 1.0,
    max_tokens: Optional[int] = None,
) -> float:
    """
    Compute reward for a math problem rollout.
    Matches PipelineRL's reward computation logic.
    
    Args:
        rollout: Rollout dictionary with 'text', 'finished', and optionally 'tokens' fields
        problem: Problem dictionary with 'answer' field
        reward_config: Reward configuration dict with reward table values
        discount_factor: Discount factor to apply based on output length (default: 1.0 = no discount)
        max_tokens: Maximum tokens for length penalty calculation
        
    Returns:
        Reward value
    """
    # Default reward table (matches PipelineRL's base.yaml)
    default_rewards = {
        "wrong_answer_not_finished": -1.0,
        "wrong_answer_finished": -0.5,
        "no_answer_not_finished": -1.0,
        "no_answer_finished": -1.0,
        "unparsable_not_finished": -1.0,
        "unparsable_finished": -1.0,
        "correct_answer_not_finished": -1.0,
        "correct_answer_finished": 1.0,
        "buffer_tokens": 0,  # 0 means no overlong reward shaping
    }
    
    # Merge with provided config
    if reward_config:
        default_rewards.update(reward_config)
    
    rewards = default_rewards
    
    generated_text = rollout.get("text", "")
    gold_answer = problem.get("answer", "")
    finished = rollout.get("finished", True)  # Default to True if not specified
    
    if not gold_answer:
        logger.warning("No gold answer provided, returning 0.0 reward")
        return 0.0
    
    # Verify answer
    answer_status = verify_math_answer(generated_text, gold_answer, strict=True)
    
    # Determine reward based on answer status and finished state (matches PipelineRL)
    reward_key = f"{answer_status}_finished" if finished else f"{answer_status}_not_finished"
    reward = rewards.get(reward_key, 0.0)
    
    # Apply discount factor based on output length (matches PipelineRL)
    output_length_tokens = len(rollout.get("tokens", []))
    if output_length_tokens > 0 and discount_factor != 1.0:
        reward *= discount_factor ** output_length_tokens
    
    # Apply length penalty (matches PipelineRL)
    overlong_penalty = 0.0
    if rewards.get("buffer_tokens", 0) > 0 and max_tokens is not None:
        overlong_penalty = length_penalty(
            max_tokens,
            output_length_tokens,
            rewards["buffer_tokens"]
        )
        reward += overlong_penalty
    
    # Store answer status and penalty in rollout for debugging
    rollout["answer_status"] = answer_status
    rollout["overlong_penalty"] = overlong_penalty
    
    return reward
