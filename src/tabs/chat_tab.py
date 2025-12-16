"""
Chat tab.

Interactive chat interface with model selection.
"""

from typing import Any

import streamlit as st

from ..inference import SamplingParams


def _get_sampling_params() -> SamplingParams:
    """Get sampling params from session state."""
    sp = st.session_state.sampling_params
    return SamplingParams(
        max_tokens=sp.get("max_tokens", 2048),
        temperature=sp.get("temperature", 1.0),
        top_p=sp.get("top_p", 0.9),
        n=1,  # Single sample for chat
        seed=sp.get("seed"),
        skip_special_tokens=sp.get("skip_special_tokens", False),
    )


def _create_conversation(model_id: str | None = None, name: str | None = None) -> str:
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
        "history": [],
        "system_prompt": "",
    }
    return conv_id


def _render_chat_messages(conv_id: str, conv: dict[str, Any]) -> None:
    """Render the message history for a conversation."""
    model_id = conv["model_id"]
    mm = st.session_state.managed_models.get(model_id)
    model_name = mm.config.name if mm else "Unknown"

    for i, msg in enumerate(conv["history"]):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**[{model_name}]** {msg['content']}")


def _generate_response(conv_id: str, conv: dict[str, Any]) -> None:
    """Generate a response for the conversation."""
    model_id = conv["model_id"]
    mm = st.session_state.managed_models.get(model_id)

    if mm is None:
        st.error("Model not found")
        return

    inference = st.session_state.inference
    tokenizer = inference.get_tokenizer(mm.config.tokenizer_id)

    messages = []
    if conv.get("system_prompt"):
        messages.append({"role": "system", "content": conv["system_prompt"]})
    messages.extend(conv["history"])

    prompt_tokens = tokenizer.apply_chat_template(
        messages,
        add_special_tokens=True,
        add_generation_prompt=True,
    )

    params = _get_sampling_params()

    with st.spinner("Generating..."):
        result = inference.sample_from_tokens(mm, prompt_tokens, params)

    response = result["results"][0]
    conv["history"].append({"role": "assistant", "content": response})


def _render_conversation(conv_id: str, conv: dict[str, Any]) -> None:
    """Render a single conversation."""
    col1, col2 = st.columns([3, 1])

    with col1:
        name = st.text_input(
            "Name",
            value=conv["name"],
            key=f"conv_name_{conv_id}",
            label_visibility="collapsed",
        )
        if name != conv["name"]:
            conv["name"] = name

    with col2:
        model_options = list(st.session_state.managed_models.keys())
        model_names = {
            mid: mm.config.name
            for mid, mm in st.session_state.managed_models.items()
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

    system_prompt = st.text_input(
        "System Prompt",
        value=conv.get("system_prompt", ""),
        key=f"conv_sys_{conv_id}",
        placeholder="Optional system prompt...",
    )
    if system_prompt != conv.get("system_prompt", ""):
        conv["system_prompt"] = system_prompt

    _render_chat_messages(conv_id, conv)

    user_input = st.chat_input("Message...", key=f"chat_input_{conv_id}")

    if user_input:
        conv["history"].append({"role": "user", "content": user_input})
        _generate_response(conv_id, conv)
        st.rerun(scope="fragment")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear History", key=f"clear_{conv_id}"):
            conv["history"] = []
            st.rerun(scope="fragment")
    with col2:
        if st.button("Delete Chat", key=f"delete_{conv_id}"):
            del st.session_state.conversations[conv_id]
            st.rerun(scope="app")


@st.fragment
def render_chat_tab() -> None:
    """Render the Chat tab."""
    if not st.session_state.managed_models:
        st.warning("No models configured. Add models in the Models tab.")
        return

    if st.button("New Chat", type="primary"):
        _create_conversation()
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
