"""
Math dataset loading utilities.
"""

import logging
from typing import List, Dict
from datasets import load_dataset
import datasets

logger = logging.getLogger(__name__)


def process_math(dataset, dataset_name: str):
    """Process math dataset items."""
    for item in dataset:
        if "problem" in item:
            question = item["problem"]
        elif "question" in item:
            question = item["question"]
        else:
            continue
        
        if "answer" in item:
            answer = f"\\boxed{{{item['answer']}}}"
        elif "solution" in item:
            answer = item["solution"]
        else:
            continue
        
        yield {
            "dataset": dataset_name,
            "task": question,
            "answer": answer,
        }


def add_ids(dataset: List[Dict]):
    """Add IDs to dataset."""
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_math_datasets(dataset_name: str, split: str = "test") -> List[Dict]:
    """
    Load math datasets from EleutherAI/hendrycks_math.
    
    Args:
        dataset_names: List of dataset names to load (should contain "math_train" or "math_test")
        split: Dataset split to load ("train" or "test")
        
    Returns:
        List of problem dictionaries with 'task' and 'answer' fields
    """
    if not dataset_name:
        raise ValueError("No dataset names provided")
    
    result_datasets = []
    
    # Use EleutherAI/hendrycks_math for train/test splits
    # This combines all math subdomains (algebra, geometry, etc.)
    if dataset_name == "hendrycks_math" and split == "train":
        data = []
        for config in [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]:
            dataset = load_dataset("EleutherAI/hendrycks_math", config, split="train")
            for sample in dataset:
                data.append(sample)
        combined_dataset = datasets.Dataset.from_list(data)
        samples = [s for s in process_math(combined_dataset, "math_train") if s is not None]
        logger.info(f"Loading math_train dataset: {len(samples)} samples")
        result_datasets.extend(add_ids(samples))
    elif dataset_name == "hendrycks_math" and split == "test":
        data = []
        for config in [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]:
            dataset = load_dataset("EleutherAI/hendrycks_math", config, split="test")
            for sample in dataset:
                data.append(sample)
        combined_dataset = datasets.Dataset.from_list(data)
        samples = [s for s in process_math(combined_dataset, "math_test") if s is not None]
        logger.info(f"Loading math_test dataset: {len(samples)} samples")
        result_datasets.extend(add_ids(samples))
    
    if len(result_datasets) == 0:
        raise ValueError(f"No datasets loaded from {dataset_name} with split={split}")
    
    logger.info(f"Total loaded: {len(result_datasets)} problems")
    return result_datasets


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    print("=" * 60)
    print("Testing Math Dataset Loading")
    print("=" * 60)
    
    try:
        # Test loading train set
        print("\n[1] Loading TRAIN set...")
        train_data = load_math_datasets(["math_train"], split="train")
        print(f"✓ Train set loaded successfully!")
        print(f"  Train set size: {len(train_data)} problems")
        
        # Test loading test set
        print("\n[2] Loading TEST set...")
        test_data = load_math_datasets(["math_test"], split="test")
        print(f"✓ Test set loaded successfully!")
        print(f"  Test set size: {len(test_data)} problems")
        
        # Show data entry keys
        print("\n[3] Data entry keys:")
        if train_data:
            print(f"  Keys: {list(train_data[0].keys())}")
        
        # Show examples
        print("\n[4] Example entries:")
        print("\n  Train example 1:")
        if train_data:
            example = train_data[0]
            print(f"    ID: {example.get('id', 'N/A')}")
            print(f"    Dataset: {example.get('dataset', 'N/A')}")
            print(f"    Task (first 200 chars): {example.get('task', '')[:200]}...")
            print(f"    Answer: {example.get('answer', 'N/A')}")
        
        print("\n  Train example 2:")
        if len(train_data) > 1:
            example = train_data[1]
            print(f"    ID: {example.get('id', 'N/A')}")
            print(f"    Dataset: {example.get('dataset', 'N/A')}")
            print(f"    Task (first 200 chars): {example.get('task', '')[:200]}...")
            print(f"    Answer: {example.get('answer', 'N/A')}")
        
        print("\n  Test example 1:")
        if test_data:
            example = test_data[0]
            print(f"    ID: {example.get('id', 'N/A')}")
            print(f"    Dataset: {example.get('dataset', 'N/A')}")
            print(f"    Task (first 200 chars): {example.get('task', '')[:200]}...")
            print(f"    Answer: {example.get('answer', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
