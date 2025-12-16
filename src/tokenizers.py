"""
Placeholder tokenizer module.

Will be replaced by character.utils.tokenizers when integrated into character repo.
"""

from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def get_tokenizer(model_id: str) -> AutoTokenizer:
    """
    Get a HuggingFace tokenizer for the given model.

    Args:
        model_id: HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B")

    Returns:
        The AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
