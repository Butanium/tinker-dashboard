"""
Chat tab.

Interactive chat interface with model selection and message editing.
"""

from datetime import datetime
from typing import Any

import streamlit as st

from ..inference import SamplingParams
from ..dashboard_state import GenerationLog, save_generation_log


def _get_sampling_params(n: int = 1) -> SamplingParams:
    """Get sampling params from session state."""
    sp = st.session_state.sampling_params
    return SamplingParams(
        max_tokens=sp.get("max_tokens", 2048),
        temperature=sp.get("temperature", 1.0),
        top_p=sp.get("top_p", 0.9),
        n=n,
        seed=sp.get("seed"),
        skip_special_tokens=sp.get("skip_special_tokens", False),
    )


def _save_conversation(conv_id: str) -> None:
    """Save conversation to cache."""
    from ..dashboard_state import save_conversation

    conv = st.session_state.conversations.get(conv_id)
    if conv:
        save_conversation(st.session_state.cache_dir / "conversations", conv_id, conv)


def _save_all_conversations() -> None:
    """Save all conversations."""
    for conv_id in st.session_state.conversations:
        _save_conversation(conv_id)


def _log_chat_generation(
    mm,
    prompt_tokens: list[int],
    outputs: list[str],
    params: SamplingParams,
    system_prompt: str = "",
    messages: list[dict] | None = None,
) -> None:
    """Log a chat generation to disk."""
    logs_dir = st.session_state.cache_dir / "generation_logs"
    log = GenerationLog(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        generation_type="chat",
        prompt_text=messages[-1]["content"] if messages else "",
        prompt_tokens=prompt_tokens,
        model_name=mm.config.name,
        sampler_path=mm.config.sampler_path,
        sampling_params={
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "n": params.n,
            "seed": params.seed,
        },
        outputs=outputs,
        system_prompt=system_prompt,
        messages=messages or [],
    )
    save_generation_log(logs_dir, log)


def create_conversation(
    model_id: str | None = None,
    name: str | None = None,
    history: list[dict] | None = None,
    system_prompt: str = "",
) -> str:
    """Create a new conversation and return its ID."""
    conv_id = f"conv_{st.session_state.conversation_counter}"
    st.session_state.conversation_counter += 1

    if model_id is None:
        active_models = [
            mm for mm in st.session_state.managed_models.values() if mm.active
        ]
        model_id = active_models[0].model_id if active_models else None

    conv_name = name or f"Chat {st.session_state.conversation_counter}"

    st.session_state.conversations[conv_id] = {
        "name": conv_name,
        "model_id": model_id,
        "history": history or [],
        "system_prompt": system_prompt,
        "editing_idx": None,
    }
    _save_conversation(conv_id)
    return conv_id


def _generate_response(
    conv_id: str,
    conv: dict[str, Any],
    continue_from_idx: int | None = None,
) -> str:
    """
    Generate a response for the conversation.

    Args:
        conv_id: Conversation ID
        conv: Conversation dict
        continue_from_idx: If set, continue from this message index

    Returns:
        Generated response text
    """
    model_id = conv["model_id"]
    mm = st.session_state.managed_models.get(model_id)

    assert mm is not None, f"Model not found: {model_id}"

    inference = st.session_state.inference
    tokenizer = inference.get_tokenizer(mm.config.base_model)

    messages = []
    if conv.get("system_prompt"):
        messages.append({"role": "system", "content": conv["system_prompt"]})

    if continue_from_idx is not None:
        messages.extend(conv["history"][: continue_from_idx + 1])
        add_gen_prompt = conv["history"][continue_from_idx]["role"] == "user"
        continue_final = not add_gen_prompt
    else:
        messages.extend(conv["history"])
        add_gen_prompt = True
        continue_final = False

    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        add_special_tokens=True,
        add_generation_prompt=add_gen_prompt,
        continue_final_message=continue_final,
    )

    params = _get_sampling_params(n=1)
    result = inference.sample_from_tokens(
        mm, prompt_tokens, params, skip_last_token=True
    )
    response = result["results"][0]

    _log_chat_generation(
        mm=mm,
        prompt_tokens=prompt_tokens,
        outputs=[response],
        params=params,
        system_prompt=conv.get("system_prompt", ""),
        messages=messages,
    )

    return response


def _handle_multi_sample(conv_id: str, conv: dict[str, Any]) -> None:
    """Handle multi-sample mode: generate N samples from the same model."""
    model_id = conv["model_id"]
    mm = st.session_state.managed_models.get(model_id)

    if mm is None:
        st.error("Model not found")
        conv.pop("pending_samples", None)
        _save_conversation(conv_id)
        return

    n_samples = st.session_state.sampling_params.get("n", 4)

    if "cached_samples" not in conv:
        inference = st.session_state.inference
        tokenizer = inference.get_tokenizer(mm.config.base_model)

        messages = []
        if conv.get("system_prompt"):
            messages.append({"role": "system", "content": conv["system_prompt"]})
        messages.extend(conv["history"])

        prompt_tokens = tokenizer.apply_chat_template(
            messages,
            add_special_tokens=True,
            add_generation_prompt=True,
        )

        params = _get_sampling_params(n=n_samples)

        with st.spinner(f"Generating {n_samples} samples from {mm.config.name}..."):
            result = inference.sample_from_tokens(mm, prompt_tokens, params)

        conv["cached_samples"] = result["results"]
        _log_chat_generation(
            mm=mm,
            prompt_tokens=prompt_tokens,
            outputs=result["results"],
            params=params,
            system_prompt=conv.get("system_prompt", ""),
            messages=messages,
        )
        _save_conversation(conv_id)

    samples = conv["cached_samples"]
    st.markdown(f"### Select one of {len(samples)} samples from **{mm.config.name}**:")

    cols = st.columns(2)
    for idx, sample in enumerate(samples):
        col_idx = idx % 2
        with cols[col_idx]:
            with st.expander(f"Sample {idx + 1}", expanded=True):
                st.markdown(sample)
                if st.button(
                    f"Use sample {idx + 1}", key=f"use_sample_{conv_id}_{idx}"
                ):
                    conv["history"].append(
                        {
                            "role": "assistant",
                            "content": sample,
                            "model_id": mm.model_id,
                        }
                    )
                    conv.pop("pending_samples", None)
                    conv.pop("cached_samples", None)
                    _save_conversation(conv_id)
                    st.rerun(scope="fragment")

    if st.button("Cancel", key=f"cancel_samples_{conv_id}"):
        conv["history"].pop()
        conv.pop("pending_samples", None)
        conv.pop("cached_samples", None)
        _save_conversation(conv_id)
        st.rerun(scope="fragment")


def _handle_multi_model(conv_id: str, conv: dict[str, Any]) -> None:
    """Handle multi-model mode: generate from all active models."""
    active_models = [mm for mm in st.session_state.managed_models.values() if mm.active]

    if not active_models:
        st.error("No active models")
        conv.pop("pending_multi_model", None)
        _save_conversation(conv_id)
        return

    if "cached_model_samples" not in conv:
        inference = st.session_state.inference
        params = _get_sampling_params(n=1)

        results = []
        with st.spinner(f"Generating from {len(active_models)} models..."):
            for mm in active_models:
                tokenizer = inference.get_tokenizer(mm.config.base_model)
                messages = []
                if conv.get("system_prompt"):
                    messages.append(
                        {"role": "system", "content": conv["system_prompt"]}
                    )
                messages.extend(conv["history"])

                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_special_tokens=True,
                    add_generation_prompt=True,
                )

                result = inference.sample_from_tokens(mm, prompt_tokens, params)
                _log_chat_generation(
                    mm=mm,
                    prompt_tokens=prompt_tokens,
                    outputs=result["results"],
                    params=params,
                    system_prompt=conv.get("system_prompt", ""),
                    messages=messages,
                )
                results.append(
                    {
                        "model_id": mm.model_id,
                        "name": mm.config.name,
                        "response": result["results"][0],
                    }
                )

        conv["cached_model_samples"] = results
        _save_conversation(conv_id)

    results = conv["cached_model_samples"]
    st.markdown(f"### Select response from one of {len(results)} models:")

    cols = st.columns(2)
    for idx, result_data in enumerate(results):
        col_idx = idx % 2
        with cols[col_idx]:
            with st.expander(f"{result_data['name']}", expanded=idx == 0):
                st.markdown(result_data["response"])
                if st.button(f"Use this", key=f"use_model_{conv_id}_{idx}"):
                    conv["history"].append(
                        {
                            "role": "assistant",
                            "content": result_data["response"],
                            "model_id": result_data["model_id"],
                        }
                    )
                    conv["model_id"] = result_data["model_id"]
                    conv.pop("pending_multi_model", None)
                    conv.pop("cached_model_samples", None)
                    _save_conversation(conv_id)
                    st.rerun(scope="fragment")

    if st.button("Cancel", key=f"cancel_models_{conv_id}"):
        conv["history"].pop()
        conv.pop("pending_multi_model", None)
        conv.pop("cached_model_samples", None)
        _save_conversation(conv_id)
        st.rerun(scope="fragment")


def _render_message_actions(
    conv_id: str,
    conv: dict[str, Any],
    idx: int,
    msg: dict,
) -> None:
    """Render action buttons for a message."""
    cols = st.columns([10, 1, 1, 1, 1])

    with cols[1]:
        if st.button("Edit", key=f"edit_{conv_id}_{idx}", help="Edit"):
            conv["editing_idx"] = idx
            st.rerun(scope="fragment")

    with cols[2]:
        if st.button("Cont", key=f"cont_{conv_id}_{idx}", help="Continue"):
            with st.spinner("Continuing..."):
                response = _generate_response(conv_id, conv, continue_from_idx=idx)
            conv["history"][idx]["content"] += response
            _save_conversation(conv_id)
            st.rerun(scope="fragment")

    with cols[3]:
        if st.button("Regen", key=f"regen_{conv_id}_{idx}", help="Regenerate"):
            conv["history"] = conv["history"][:idx]
            with st.spinner("Regenerating..."):
                response = _generate_response(conv_id, conv)
            conv["history"].append(
                {"role": "assistant", "content": response, "model_id": conv["model_id"]}
            )
            _save_conversation(conv_id)
            st.rerun(scope="fragment")

    with cols[4]:
        if st.button("Del", key=f"del_{conv_id}_{idx}", help="Delete"):
            conv["history"].pop(idx)
            _save_conversation(conv_id)
            st.rerun(scope="fragment")


def _render_chat_messages(conv_id: str, conv: dict[str, Any]) -> None:
    """Render the message history for a conversation with action buttons."""
    editing_idx = conv.get("editing_idx")

    for i, msg in enumerate(conv["history"]):
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            if editing_idx == i:
                edited = st.text_area(
                    "Edit message",
                    value=content,
                    key=f"edit_area_{conv_id}_{i}",
                    label_visibility="collapsed",
                    height=150,
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Save", key=f"save_edit_{conv_id}_{i}", type="primary"
                    ):
                        conv["history"][i]["content"] = edited
                        conv["editing_idx"] = None
                        _save_conversation(conv_id)
                        st.rerun(scope="fragment")
                with col2:
                    if st.button("Cancel", key=f"cancel_edit_{conv_id}_{i}"):
                        conv["editing_idx"] = None
                        st.rerun(scope="fragment")
            else:
                if role == "assistant":
                    msg_model_id = msg.get("model_id", conv["model_id"])
                    mm = st.session_state.managed_models.get(msg_model_id)
                    model_name = mm.config.name if mm else "Unknown"
                    st.markdown(f"**[{model_name}]** {content}")
                else:
                    st.markdown(content)
                _render_message_actions(conv_id, conv, i, msg)


def _render_conversation(conv_id: str, conv: dict[str, Any]) -> None:
    """Render a single conversation."""
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        name = st.text_input(
            "Name",
            value=conv["name"],
            key=f"conv_name_{conv_id}",
            label_visibility="collapsed",
        )
        if name != conv["name"]:
            conv["name"] = name
            _save_conversation(conv_id)

    with col2:
        model_options = list(st.session_state.managed_models.keys())
        model_names = {
            mid: mm.config.name for mid, mm in st.session_state.managed_models.items()
        }
        current_idx = (
            model_options.index(conv["model_id"])
            if conv["model_id"] in model_options
            else 0
        )
        selected = st.selectbox(
            "Model",
            options=model_options,
            index=current_idx,
            format_func=lambda x: model_names.get(x, x),
            key=f"conv_model_{conv_id}",
            label_visibility="collapsed",
        )
        if selected != conv["model_id"]:
            conv["model_id"] = selected
            _save_conversation(conv_id)

    with col3:
        gen_mode = st.selectbox(
            "Gen mode",
            options=["single", "multi-sample", "multi-model"],
            index=["single", "multi-sample", "multi-model"].index(
                conv.get("gen_mode", "single")
            ),
            key=f"conv_gen_mode_{conv_id}",
            label_visibility="collapsed",
            help="single: one sample | multi-sample: N samples from current model | multi-model: one sample from each active model",
        )
        if gen_mode != conv.get("gen_mode", "single"):
            conv["gen_mode"] = gen_mode
            _save_conversation(conv_id)

    system_prompt = st.text_input(
        "System Prompt",
        value=conv.get("system_prompt", ""),
        key=f"conv_sys_{conv_id}",
        placeholder="Optional system prompt...",
    )
    if system_prompt != conv.get("system_prompt", ""):
        conv["system_prompt"] = system_prompt
        _save_conversation(conv_id)

    _render_chat_messages(conv_id, conv)

    user_input = st.chat_input("Message...", key=f"chat_input_{conv_id}")

    if user_input:
        conv["history"].append({"role": "user", "content": user_input})
        gen_mode = conv.get("gen_mode", "single")

        if gen_mode == "multi-sample":
            conv["pending_samples"] = True
            _save_conversation(conv_id)
            st.rerun(scope="fragment")
        elif gen_mode == "multi-model":
            conv["pending_multi_model"] = True
            _save_conversation(conv_id)
            st.rerun(scope="fragment")
        else:
            with st.spinner("Generating..."):
                response = _generate_response(conv_id, conv)
            conv["history"].append(
                {"role": "assistant", "content": response, "model_id": conv["model_id"]}
            )
            _save_conversation(conv_id)
            st.rerun(scope="fragment")

    if conv.get("pending_samples"):
        _handle_multi_sample(conv_id, conv)
        return

    if conv.get("pending_multi_model"):
        _handle_multi_model(conv_id, conv)
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear History", key=f"clear_{conv_id}"):
            conv["history"] = []
            _save_conversation(conv_id)
            st.rerun(scope="fragment")
    with col2:
        if st.button("Delete Chat", key=f"delete_{conv_id}"):
            from ..dashboard_state import delete_conversation

            delete_conversation(st.session_state.cache_dir / "conversations", conv_id)
            del st.session_state.conversations[conv_id]
            st.rerun(scope="app")


@st.fragment
def render_chat_tab() -> None:
    """Render the Chat tab."""
    if not st.session_state.managed_models:
        st.warning("No models configured. Add models in the Models tab.")
        return

    if st.button("New Chat", type="primary"):
        create_conversation()
        st.rerun(scope="fragment")

    if not st.session_state.conversations:
        st.info("No conversations yet. Click 'New Chat' to start.")
        return

    conv_items = list(st.session_state.conversations.items())
    tab_names = [conv["name"] for _, conv in conv_items]
    tabs = st.tabs(tab_names)

    for tab, (conv_id, conv) in zip(tabs, conv_items):
        with tab:
            _render_conversation(conv_id, conv)
