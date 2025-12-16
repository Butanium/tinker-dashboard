from dotenv import load_dotenv
from pathlib import Path

import tinker
from tinker import types

from character.utils.tokenizers import get_tokenizer

env_path = Path("~/tinker-cookbook/.env").expanduser()
load_dotenv(dotenv_path=env_path)

service_cl = tinker.ServiceClient()


def get_response(
    model: str,
    sampler_path: str,
    prompt: str,
) -> str:
    sampling_cl = service_cl.create_sampling_client(sampler_path)

    if model == "qwen":
        tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    elif model == "llama":
        tokenizer = get_tokenizer("meta-llama/Llama-3.3-70B-Instruct")
    else:
        raise ValueError(f"Unknown model: {model}")

    messages = [
        {"role": "user", "content": prompt},
    ]
    tkns = tokenizer.apply_chat_template(
        messages, add_special_tokens=True, add_generation_prompt=True
    )
    prompt = types.ModelInput.from_ints(tkns)
    params = types.SamplingParams(max_tokens=2048, temperature=1.0)
    future = sampling_cl.sample(prompt=prompt, sampling_params=params, num_samples=1)
    result = future.result()
    return tokenizer.decode(result.sequences[0].tokens)


llama_math = (
    "tinker://9f03afaa-7194-578c-a86c-b0cd5eae8411:train:0/sampler_weights/final"
)
out = get_response(
    "llama", llama_math, "What would you say are your main goals and drives?"
)
print(out)
