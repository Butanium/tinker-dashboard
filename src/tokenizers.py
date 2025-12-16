from functools import cache
from transformers import AutoTokenizer


@cache
def get_tokenizer(model_id: str) -> AutoTokenizer:
    """
    Get a HuggingFace tokenizer for the given model.

    Args:
        model_id: HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B")

    Returns:
        The AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
