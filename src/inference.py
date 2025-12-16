"""
Tinker inference wrapper for the dashboard.
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import tinker
from tinker import types

from .tokenizers import get_tokenizer
from .model_config import ManagedModel


@dataclass
class SamplingParams:
    """Sampling parameters for generation."""

    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    n: int = 1
    seed: int | None = None
    skip_special_tokens: bool = False


class TinkerInference:
    """Wrapper for tinker sampling service."""

    def __init__(self, env_path: Path | None = None):
        """
        Initialize the inference client.

        Args:
            env_path: Path to .env file. Defaults to ~/tinker-cookbook/.env
        """
        if env_path is None:
            env_path = Path("~/tinker-cookbook/.env").expanduser()
        load_dotenv(dotenv_path=env_path)
        self._service_cl = tinker.ServiceClient()
        self._sampling_clients: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

    def get_tokenizer(self, tokenizer_id: str):
        """Get or create a cached tokenizer."""
        if tokenizer_id not in self._tokenizers:
            self._tokenizers[tokenizer_id] = get_tokenizer(tokenizer_id)
        return self._tokenizers[tokenizer_id]

    def _get_sampling_client(self, sampler_path: str):
        """Get or create a cached sampling client."""
        if sampler_path not in self._sampling_clients:
            self._sampling_clients[sampler_path] = (
                self._service_cl.create_sampling_client(sampler_path)
            )
        return self._sampling_clients[sampler_path]

    def sample(
        self,
        sampler_path: str,
        tokenizer_id: str,
        prompt_tokens: list[int],
        params: SamplingParams,
    ) -> list[str]:
        """
        Sample from a model and decode with the specified tokenizer.

        Args:
            sampler_path: tinker:// URI to the sampler
            tokenizer_id: HuggingFace tokenizer ID for decoding
            prompt_tokens: Tokenized prompt
            params: Sampling parameters

        Returns:
            List of decoded text samples
        """
        sampling_cl = self._get_sampling_client(sampler_path)
        tokenizer = self.get_tokenizer(tokenizer_id)

        prompt = types.ModelInput.from_ints(prompt_tokens)
        tinker_params = types.SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )

        future = sampling_cl.sample(
            prompt=prompt,
            sampling_params=tinker_params,
            num_samples=params.n,
        )
        result = future.result()

        samples = []
        for seq in result.sequences:
            text = tokenizer.decode(
                seq.tokens, skip_special_tokens=params.skip_special_tokens
            )
            samples.append(text)

        return samples

    def multi_model_sample(
        self,
        models: list[ManagedModel],
        prompt_text: str,
        params: SamplingParams,
        system_prompt: str = "",
        assistant_prefill: str = "",
    ):
        """
        Sample from multiple models with the same prompt.

        Yields results as they complete for progressive rendering.

        Args:
            models: List of ManagedModel instances
            prompt_text: User message text
            params: Sampling parameters
            system_prompt: Optional system prompt
            assistant_prefill: Optional assistant prefill

        Yields:
            dict with {model, results, prompt_tokens}
        """
        for mm in models:
            tokenizer = self.get_tokenizer(mm.config.tokenizer_id)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})

            if assistant_prefill:
                messages.append({"role": "assistant", "content": assistant_prefill})
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_special_tokens=True,
                    continue_final_message=True,
                )
            else:
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_special_tokens=True,
                    add_generation_prompt=True,
                )

            results = self.sample(
                sampler_path=mm.config.sampler_path,
                tokenizer_id=mm.config.tokenizer_id,
                prompt_tokens=prompt_tokens,
                params=params,
            )

            yield {
                "model": mm,
                "results": results,
                "prompt_tokens": prompt_tokens,
            }

    def sample_from_tokens(
        self,
        mm: ManagedModel,
        prompt_tokens: list[int],
        params: SamplingParams,
    ) -> dict:
        """
        Sample from a single model with pre-tokenized prompt.

        Args:
            mm: ManagedModel instance
            prompt_tokens: Pre-tokenized prompt
            params: Sampling parameters

        Returns:
            dict with {model, results, prompt_tokens}
        """
        results = self.sample(
            sampler_path=mm.config.sampler_path,
            tokenizer_id=mm.config.tokenizer_id,
            prompt_tokens=prompt_tokens,
            params=params,
        )
        return {
            "model": mm,
            "results": results,
            "prompt_tokens": prompt_tokens,
        }
