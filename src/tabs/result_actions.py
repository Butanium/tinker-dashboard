"""
Shared result action buttons for multi-gen and multi-prompt tabs.
"""

import streamlit as st

from ..inference import SamplingParams


def _get_single_sample_params() -> SamplingParams:
    """Get sampling params with n=1 for continue operations."""
    sp = st.session_state.sampling_params
    return SamplingParams(
        max_tokens=sp.get("max_tokens", 2048),
        temperature=sp.get("temperature", 1.0),
        top_p=sp.get("top_p", 0.9),
        n=1,
        seed=sp.get("seed"),
        skip_special_tokens=sp.get("skip_special_tokens", False),
    )


def render_result_actions(
    result_data: dict,
    prompt_info: dict,
    key_prefix: str,
    on_update: callable,
) -> None:
    """
    Render action buttons for a single model's results.

    Args:
        result_data: Dict with 'model', 'results', 'prompt_tokens' keys
        prompt_info: Dict with prompt context - either:
            - {'prompt': str, 'system_prompt': str} for text mode
            - {'messages': list, 'system_prompt': str} for messages mode
            - {'managed_prompt': ManagedPrompt} for multi-prompt mode
        key_prefix: Unique prefix for button keys
        on_update: Callback to trigger UI refresh after modification
    """
    mm = result_data["model"]
    samples = result_data["results"]
    inference = st.session_state.inference
    tokenizer = inference.get_tokenizer(mm.config.base_model)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Regenerate", key=f"{key_prefix}_regen", use_container_width=True):
            params = _get_sampling_params_full()
            prompt_tokens = _build_prompt_tokens(prompt_info, tokenizer)
            new_result = inference.sample_from_tokens(mm, prompt_tokens, params)
            result_data["results"] = new_result["results"]
            result_data["prompt_tokens"] = prompt_tokens
            on_update()

    with col2:
        if st.button("Continue", key=f"{key_prefix}_cont", use_container_width=True):
            params = _get_single_sample_params()
            new_samples = []
            for sample in samples:
                cont_tokens = _build_continue_tokens(prompt_info, sample, tokenizer)
                cont_result = inference.sample_from_tokens(mm, cont_tokens, params)
                new_samples.append(sample + cont_result["results"][0])
            result_data["results"] = new_samples
            on_update()

    with col3:
        md_content = _generate_result_markdown(result_data, prompt_info)
        st.download_button(
            "Save",
            data=md_content,
            file_name=f"{mm.config.name.replace(' ', '_')}_samples.md",
            mime="text/markdown",
            key=f"{key_prefix}_save",
            use_container_width=True,
        )


def _get_sampling_params_full() -> SamplingParams:
    """Get full sampling params from session state."""
    sp = st.session_state.sampling_params
    return SamplingParams(
        max_tokens=sp.get("max_tokens", 2048),
        temperature=sp.get("temperature", 1.0),
        top_p=sp.get("top_p", 0.9),
        n=sp.get("n", 1),
        seed=sp.get("seed"),
        skip_special_tokens=sp.get("skip_special_tokens", False),
    )


def _build_prompt_tokens(prompt_info: dict, tokenizer) -> list[int]:
    """Build prompt tokens from prompt_info."""
    if "managed_prompt" in prompt_info:
        mp = prompt_info["managed_prompt"]
        return _tokenize_managed_prompt(mp, tokenizer)

    messages = []
    if prompt_info.get("system_prompt"):
        messages.append({"role": "system", "content": prompt_info["system_prompt"]})

    if "messages" in prompt_info:
        messages.extend(prompt_info["messages"])
        last_role = (
            prompt_info["messages"][-1]["role"] if prompt_info["messages"] else "user"
        )
        add_gen = last_role == "user"
        continue_final = last_role == "assistant"
    else:
        messages.append({"role": "user", "content": prompt_info.get("prompt", "")})
        add_gen = True
        continue_final = False

    if prompt_info.get("assistant_prefill"):
        messages.append(
            {"role": "assistant", "content": prompt_info["assistant_prefill"]}
        )
        add_gen = False
        continue_final = True

    return tokenizer.apply_chat_template(
        messages,
        add_special_tokens=True,
        add_generation_prompt=add_gen,
        continue_final_message=continue_final,
    )


def _build_continue_tokens(prompt_info: dict, sample: str, tokenizer) -> list[int]:
    """Build tokens for continuing a sample."""
    messages = []

    if "managed_prompt" in prompt_info:
        mp = prompt_info["managed_prompt"]
        if mp.system_prompt:
            messages.append({"role": "system", "content": mp.system_prompt})
        if mp.prompt_mode == "messages":
            messages.extend(mp.messages)
        else:
            messages.append({"role": "user", "content": mp.content})
    else:
        if prompt_info.get("system_prompt"):
            messages.append({"role": "system", "content": prompt_info["system_prompt"]})
        if "messages" in prompt_info:
            messages.extend(prompt_info["messages"])
        else:
            messages.append({"role": "user", "content": prompt_info.get("prompt", "")})

    messages.append({"role": "assistant", "content": sample})

    return tokenizer.apply_chat_template(
        messages,
        add_special_tokens=True,
        continue_final_message=True,
    )


def _tokenize_managed_prompt(mp, tokenizer) -> list[int]:
    """Tokenize a ManagedPrompt."""
    if mp.prompt_mode == "text":
        if mp.template_mode == "No template":
            return tokenizer.encode(mp.content, add_special_tokens=True)
        elif mp.template_mode == "Apply chat template":
            messages = []
            if mp.system_prompt:
                messages.append({"role": "system", "content": mp.system_prompt})
            messages.append({"role": "user", "content": mp.content})
            return tokenizer.apply_chat_template(
                messages,
                add_special_tokens=True,
                add_generation_prompt=True,
            )
        elif mp.template_mode == "Apply loom template":
            return tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "The assistant is in CLI simulation mode.",
                    },
                    {"role": "user", "content": "<cmd>cat untitled.txt</cmd>"},
                    {"role": "assistant", "content": mp.content},
                ],
                continue_final_message=True,
            )
    else:
        all_messages = []
        if mp.system_prompt:
            all_messages.append({"role": "system", "content": mp.system_prompt})
        all_messages.extend(mp.messages)

        assert all_messages, f"Prompt '{mp.name}' has no messages"

        last_role = mp.messages[-1]["role"] if mp.messages else "system"
        add_gen = last_role in ["user", "system"]
        continue_final = last_role == "assistant"

        return tokenizer.apply_chat_template(
            all_messages,
            add_special_tokens=True,
            add_generation_prompt=add_gen,
            continue_final_message=continue_final,
        )


def _generate_result_markdown(result_data: dict, prompt_info: dict) -> str:
    """Generate markdown for a single model's results."""
    mm = result_data["model"]
    samples = result_data["results"]

    md = f"# {mm.config.name}\n\n"

    if "managed_prompt" in prompt_info:
        mp = prompt_info["managed_prompt"]
        if mp.prompt_mode == "messages":
            md += "**Messages:**\n"
            for msg in mp.messages:
                md += f"- {msg['role']}: {msg['content']}\n"
            md += "\n"
        else:
            md += f"**Prompt:** {mp.content}\n\n"
        if mp.system_prompt:
            md += f"**System:** {mp.system_prompt}\n\n"
    else:
        if "messages" in prompt_info:
            md += "**Messages:**\n"
            for msg in prompt_info["messages"]:
                md += f"- {msg['role']}: {msg['content']}\n"
            md += "\n"
        else:
            md += f"**Prompt:** {prompt_info.get('prompt', '')}\n\n"
        if prompt_info.get("system_prompt"):
            md += f"**System:** {prompt_info['system_prompt']}\n\n"

    md += "---\n\n"
    for idx, sample in enumerate(samples):
        if len(samples) > 1:
            md += f"## Sample {idx + 1}\n\n"
        md += f"{sample}\n\n"

    return md


def render_decoded_prompt(
    result_data: dict,
    key_prefix: str,
    expanded: bool = False,
) -> None:
    """
    Render decoded prompt with special tokens for a single result.

    Args:
        result_data: Dict with 'model', 'prompt_tokens' keys
        key_prefix: Unique prefix for expander key
        expanded: Whether expander is initially expanded
    """
    mm = result_data["model"]
    inference = st.session_state.inference
    tokenizer = inference.get_tokenizer(mm.config.base_model)
    decoded = tokenizer.decode(result_data["prompt_tokens"], skip_special_tokens=False)

    with st.expander(f"Prompt ({mm.config.name})", expanded=expanded):
        st.code(decoded, language=None)


def render_sample_actions(
    samples: list[str],
    sample_idx: int,
    mm,
    prompt_info: dict,
    key_prefix: str,
    on_update: callable,
) -> None:
    """
    Render action buttons for a single sample within a list.

    Used in chat_tab multi-sample mode where we have N samples from one model.

    Args:
        samples: The list of samples (will be mutated)
        sample_idx: Index of the current sample
        mm: ManagedModel instance
        prompt_info: Dict with prompt context (same as render_result_actions)
        key_prefix: Unique prefix for button keys
        on_update: Callback to trigger UI refresh after modification
    """
    inference = st.session_state.inference
    tokenizer = inference.get_tokenizer(mm.config.base_model)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Regen", key=f"{key_prefix}_regen", use_container_width=True):
            params = _get_single_sample_params()
            prompt_tokens = _build_prompt_tokens(prompt_info, tokenizer)
            result = inference.sample_from_tokens(mm, prompt_tokens, params)
            samples[sample_idx] = result["results"][0]
            on_update()

    with col2:
        if st.button("Cont", key=f"{key_prefix}_cont", use_container_width=True):
            params = _get_single_sample_params()
            cont_tokens = _build_continue_tokens(
                prompt_info, samples[sample_idx], tokenizer
            )
            result = inference.sample_from_tokens(mm, cont_tokens, params)
            samples[sample_idx] = samples[sample_idx] + result["results"][0]
            on_update()

    with col3:
        sample = samples[sample_idx]
        md = f"# Sample {sample_idx + 1}\n\n{sample}\n"
        st.download_button(
            "Save",
            data=md,
            file_name=f"sample_{sample_idx + 1}.md",
            mime="text/markdown",
            key=f"{key_prefix}_save",
            use_container_width=True,
        )
