"""
Training utilities for RL training.
Model loading, tokenization, collation, and RL training step.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Mask value for prompt tokens (not used in loss computation)
MASKED_TOKEN_ID = -100


def load_model_and_tokenizer(cfg: DictConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from config.
    
    Args:
        cfg: Configuration with model_path
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = cfg.model_path
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (don't use device_map for single GPU - we'll move manually)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled to save memory")
    
    model.train()
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    return model, tokenizer


def tokenize_sample(
    sample: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    system_prompt: str,
    task_template: str,
) -> Dict[str, Any]:
    """Tokenize a sample for training.
    
    Args:
        sample: Sample dict with 'text', 'tokens', 'logprobs', 'problem', 'reward', 'ref_logprobs'
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        system_prompt: System prompt string
        task_template: Task template string (e.g., "{task}")
        
    Returns:
        Tokenized dict with 'input_ids', 'labels', 'attention_mask', 'reward', 'logprobs', 'ref_logprobs'
    """
    problem = sample.get("problem", {})
    task = problem.get("task", "")
    generated_text = sample.get("text", "")
    
    # Build messages (same format as used in rollout)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    task_text = task_template.format(task=task)
    messages.append({"role": "user", "content": task_text})
    
    # Build prompt text (without generation)
    # Try to use chat template, fallback to simple concatenation if not available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Build full text (prompt + generation)
        full_messages = messages + [{"role": "assistant", "content": generated_text}]
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
        )
    else:
        # Fallback: simple text concatenation
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        prompt_parts.append(task_text)
        prompt_text = "\n\n".join(prompt_parts)
        full_text = prompt_text + "\n\n" + generated_text
    
    # Tokenize full text
    # If we used chat template, it already added special tokens, so don't add them again
    # Otherwise, add special tokens
    used_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    tokenizer_output = tokenizer(
        full_text,
        return_offsets_mapping=True,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=not used_chat_template,
    )
    
    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    offset_mapping = tokenizer_output.get("offset_mapping", [])
    
    # Validate token IDs are within vocabulary range
    vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
    invalid_ids = [i for i, tid in enumerate(input_ids) if tid >= vocab_size or tid < 0]
    if invalid_ids:
        logger.warning(f"Found {len(invalid_ids)} invalid token IDs in tokenization (vocab_size={vocab_size})")
        # Clamp to valid range
        input_ids = [max(0, min(tid, vocab_size - 1)) for tid in input_ids]
    
    # Find where generation starts (after prompt)
    prompt_length = len(prompt_text)
    generation_start_char = len(full_text) - len(generated_text)
    
    # Create labels: mask prompt tokens, keep generation tokens
    labels = []
    for i, (start_char, end_char) in enumerate(offset_mapping):
        if start_char >= generation_start_char:
            # This is a generation token
            token_id = input_ids[i]
            # Ensure token ID is valid
            if token_id >= vocab_size or token_id < 0:
                logger.warning(f"Invalid generation token ID {token_id} at position {i}, clamping to valid range")
                token_id = max(0, min(token_id, vocab_size - 1))
            labels.append(token_id)
        else:
            # This is a prompt token - mask it
            labels.append(MASKED_TOKEN_ID)
    
    # Get reward and logprobs from sample
    reward = sample.get("reward", 0.0)
    logprobs = sample.get("logprobs", [])
    ref_logprobs = sample.get("ref_logprobs", [])
    
    # Ensure logprobs match generation length
    generation_token_count = sum(1 for label in labels if label != MASKED_TOKEN_ID)
    if len(logprobs) != generation_token_count:
        logger.warning(
            f"Logprobs length ({len(logprobs)}) doesn't match generation tokens ({generation_token_count}). "
            f"Truncating/padding to match."
        )
        if len(logprobs) > generation_token_count:
            logprobs = logprobs[:generation_token_count]
        else:
            logprobs = logprobs + [0.0] * (generation_token_count - len(logprobs))
    
    if len(ref_logprobs) != generation_token_count:
        if len(ref_logprobs) > generation_token_count:
            ref_logprobs = ref_logprobs[:generation_token_count]
        else:
            ref_logprobs = ref_logprobs + [0.0] * (generation_token_count - len(ref_logprobs))
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "reward": reward,
        "logprobs": logprobs,
        "ref_logprobs": ref_logprobs,
    }


def collate_batch(
    tokenized_samples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    pad_to_multiple_of: int = 16,
) -> Dict[str, torch.Tensor]:
    """Collate a list of tokenized samples into a batch.
    
    Args:
        tokenized_samples: List of tokenized sample dicts
        tokenizer: Tokenizer instance (for pad_token_id)
        pad_to_multiple_of: Pad sequence length to multiple of this value
        
    Returns:
        Dict of batched tensors: input_ids, labels, attention_mask, rewards, logprobs, ref_logprobs
    """
    # Find max length in batch
    max_length = max(len(s["input_ids"]) for s in tokenized_samples)
    
    # Pad to multiple if specified
    if pad_to_multiple_of > 0:
        max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    
    batch_size = len(tokenized_samples)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Initialize tensors
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), MASKED_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    
    # Lists for per-sample values (will be converted to tensors)
    rewards = []
    logprobs_list = []
    ref_logprobs_list = []
    
    # Fill tensors
    for i, sample in enumerate(tokenized_samples):
        seq_len = len(sample["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(sample["input_ids"], dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(sample["labels"], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(sample["attention_mask"], dtype=torch.long)
        
        rewards.append(sample["reward"])
        logprobs_list.append(sample["logprobs"])
        ref_logprobs_list.append(sample["ref_logprobs"])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "rewards": rewards,  # List of floats (one per sample)
        "logprobs": logprobs_list,  # List of lists (one per sample)
        "ref_logprobs": ref_logprobs_list,  # List of lists (one per sample)
    }


def rl_step(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulate_gradients: bool = False,
    gradient_accumulation_steps: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """Perform a single RL training step.
    
    Args:
        model: Model to train
        batch: Batched tensors from collate_batch
        optimizer: Optimizer instance
        device: Device to run on
        
    Returns:
        Tuple of (loss_value, metrics_dict)
    """
    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    rewards = batch["rewards"]  # List of floats (one per sample)
    
    # Validate token IDs are within vocabulary range
    vocab_size = model.config.vocab_size
    invalid_input_ids = (input_ids >= vocab_size) | (input_ids < 0)
    if invalid_input_ids.any():
        invalid_count = invalid_input_ids.sum().item()
        max_id = input_ids.max().item()
        min_id = input_ids[input_ids >= 0].min().item() if (input_ids >= 0).any() else -1
        logger.error(f"Invalid input_ids detected: {invalid_count} tokens out of range [0, {vocab_size-1}]")
        logger.error(f"  Max ID: {max_id}, Min ID: {min_id}, Vocab size: {vocab_size}")
        # Clamp invalid IDs to valid range
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # Compute log probabilities for actual tokens
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Validate labels before gather operation
    # Only check non-masked labels (MASKED_TOKEN_ID is -100, which is fine)
    valid_mask = (shift_labels != MASKED_TOKEN_ID)
    invalid_labels = valid_mask & ((shift_labels >= vocab_size) | (shift_labels < 0))
    if invalid_labels.any():
        invalid_count = invalid_labels.sum().item()
        max_label = shift_labels[valid_mask].max().item() if valid_mask.any() else -1
        min_label = shift_labels[valid_mask & (shift_labels >= 0)].min().item() if (valid_mask & (shift_labels >= 0)).any() else -1
        logger.error(f"Invalid labels detected: {invalid_count} tokens out of range [0, {vocab_size-1}]")
        logger.error(f"  Max label: {max_label}, Min label: {min_label}, Vocab size: {vocab_size}")
        # Clamp invalid labels to valid range (but keep MASKED_TOKEN_ID as is)
        shift_labels = torch.where(
            invalid_labels,
            torch.clamp(shift_labels, 0, vocab_size - 1),
            shift_labels
        )
    
    # Get log probs for the tokens that were actually generated
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for the actual token IDs
    batch_size, seq_len, vocab_size = log_probs.shape
    shift_labels_expanded = shift_labels.unsqueeze(-1)  # [batch_size, seq_len, 1]
    
    # For masked tokens, use index 0 (will be masked out anyway)
    # This avoids CUDA assert when gathering with MASKED_TOKEN_ID (-100)
    gather_indices = torch.where(
        shift_labels == MASKED_TOKEN_ID,
        torch.zeros_like(shift_labels),
        shift_labels
    ).unsqueeze(-1)
    
    token_log_probs = torch.gather(log_probs, dim=-1, index=gather_indices).squeeze(-1)
    # [batch_size, seq_len]
    
    # Mask out non-generation tokens (labels == MASKED_TOKEN_ID)
    mask = (shift_labels != MASKED_TOKEN_ID).float()
    token_log_probs_masked = token_log_probs * mask
    
    # Expand rewards to per-token (one reward per sample)
    # rewards is a list of floats, one per sample
    rewards_tensor = torch.tensor(rewards, device=device)  # [batch_size]
    rewards_per_token = rewards_tensor.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
    
    # Compute policy gradient loss: -mean(log_prob * reward) for generation tokens only
    # We want to maximize log_prob * reward, so loss = -log_prob * reward
    loss_per_token = -token_log_probs_masked * rewards_per_token * mask
    
    # Average over generation tokens only
    num_generation_tokens = mask.sum()
    if num_generation_tokens > 0:
        loss = loss_per_token.sum() / num_generation_tokens
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Scale loss for gradient accumulation (divide by accumulation steps to average)
    # This ensures gradients are averaged, not summed, across accumulation steps
    if accumulate_gradients:
        loss = loss / gradient_accumulation_steps
    
    # Backward pass
    if not accumulate_gradients:
        # Normal mode: zero gradients before backward
        optimizer.zero_grad()
    # Otherwise, gradients accumulate across steps
    
    loss.backward()
    
    # Only step optimizer if not accumulating (or if this is the last accumulation step)
    if not accumulate_gradients:
        optimizer.step()
    
    # Compute metrics
    metrics = {
        "loss": loss.item(),
        "num_generation_tokens": num_generation_tokens.item(),
        "mean_reward": torch.tensor(rewards).mean().item(),
        "mean_log_prob": token_log_probs_masked.sum(dim=1).mean().item() / mask.sum(dim=1).mean().item() if mask.sum(dim=1).mean() > 0 else 0.0,
    }
    
    return loss.item(), metrics


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
) -> None:
    """Save training checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        optimizer: Optimizer state to save
        step: Training step number
        output_dir: Output directory
    """
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {checkpoint_dir}")
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save optimizer state
    optimizer_state_path = checkpoint_dir / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_state_path)
    
    logger.info(f"Checkpoint saved: step {step}")

