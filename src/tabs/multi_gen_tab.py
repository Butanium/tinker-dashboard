"""
Multi-generation tab.

Run multiple models on the same prompt and compare outputs side-by-side.
"""

import html
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ..inference import SamplingParams
from ..dashboard_state import GenerationLog, save_generation_log


COMPONENTS_DIR = Path(__file__).parent.parent.parent / "components"
_SAMPLE_CYCLER_JS = (COMPONENTS_DIR / "sample_cycler.js").read_text()
_SAMPLE_CYCLER_CSS = (COMPONENTS_DIR / "sample_cycler.css").read_text()
_SAMPLE_CYCLER_HTML = (COMPONENTS_DIR / "sample_cycler.html").read_text()


def render_sample_cycler(
    samples: list[str], component_id: str, height: int = 400
) -> None:
    """Render an HTML component for cycling through samples."""
    samples_html = "\n".join(
        f'<div class="sample-content" style="display: {"block" if i == 0 else "none"}">'
        f"{html.escape(s)}</div>"
        for i, s in enumerate(samples)
    )

    rendered = _SAMPLE_CYCLER_HTML
    rendered = rendered.replace("{{CSS}}", _SAMPLE_CYCLER_CSS)
    rendered = rendered.replace("{{JS}}", _SAMPLE_CYCLER_JS)
    rendered = rendered.replace("{{ID}}", component_id)
    rendered = rendered.replace("{{TOTAL}}", str(len(samples)))
    rendered = rendered.replace("{{SAMPLES}}", samples_html)

    if len(samples) > 1:
        rendered = rendered.replace("{{#if MULTI}}", "").replace("{{/if}}", "")
    else:
        rendered = re.sub(
            r"\{\{#if MULTI\}\}.*?\{\{/if\}\}", "", rendered, flags=re.DOTALL
        )

    components.html(rendered, height=height, scrolling=True)


def _get_sampling_params() -> SamplingParams:
    """Get sampling params from session state."""
    sp = st.session_state.sampling_params
    return SamplingParams(
        max_tokens=sp.get("max_tokens", 2048),
        temperature=sp.get("temperature", 1.0),
        top_p=sp.get("top_p", 0.9),
        n=sp.get("n", 1),
        seed=sp.get("seed"),
        skip_special_tokens=sp.get("skip_special_tokens", False),
    )


def _log_generation(
    prompt_text: str,
    prompt_tokens: list[int],
    mm,
    outputs: list[str],
    params: SamplingParams,
    system_prompt: str = "",
    messages: list[dict] | None = None,
) -> None:
    """Log a generation to disk."""
    logs_dir = st.session_state.cache_dir / "generation_logs"
    log = GenerationLog(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        generation_type="multigen",
        prompt_text=prompt_text,
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


def _render_import_section() -> None:
    """Render import from prompts/chats section."""
    with st.expander("Import from...", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**From Saved Prompts**")
            prompts = st.session_state.get("managed_prompts", {})
            if prompts:
                prompt_options = {
                    mp.prompt_id: mp.get_display_name() for mp in prompts.values()
                }
                selected_prompt = st.selectbox(
                    "Select prompt",
                    options=list(prompt_options.keys()),
                    format_func=lambda x: prompt_options.get(x, x),
                    key="import_prompt_select",
                    label_visibility="collapsed",
                )
                if st.button("Import Prompt", key="import_prompt_btn"):
                    mp = prompts[selected_prompt]
                    if mp.prompt_mode == "text":
                        st.session_state.multi_gen_prompt = mp.content
                        st.session_state.multi_gen_system_prompt = mp.system_prompt
                        st.session_state.multi_gen_template_mode = mp.template_mode
                    else:
                        st.session_state.multi_gen_messages = list(mp.messages)
                        st.session_state.msg_builder_system = mp.system_prompt
                    st.rerun(scope="fragment")
            else:
                st.info("No saved prompts")

        with col2:
            st.markdown("**From Chat Conversations**")
            convs = st.session_state.get("conversations", {})
            if convs:
                conv_options = {
                    cid: conv.get("name", cid) for cid, conv in convs.items()
                }
                selected_conv = st.selectbox(
                    "Select conversation",
                    options=list(conv_options.keys()),
                    format_func=lambda x: conv_options.get(x, x),
                    key="import_conv_select",
                    label_visibility="collapsed",
                )
                if st.button("Import Chat", key="import_conv_btn"):
                    conv = convs[selected_conv]
                    st.session_state.multi_gen_messages = list(conv.get("history", []))
                    st.session_state.msg_builder_system = conv.get("system_prompt", "")
                    st.rerun(scope="fragment")
            else:
                st.info("No conversations")


def _render_text_input_tab() -> None:
    """Render the text input tab."""
    template_mode = st.selectbox(
        "Template Mode",
        options=["Apply chat template", "No template", "Apply loom template"],
        key="multi_gen_template_mode",
    )

    if template_mode == "Apply chat template":
        st.text_area(
            "System Prompt (optional)",
            key="multi_gen_system_prompt",
            height=68,
        )
        st.text_input(
            "Assistant Prefill (optional)",
            key="multi_gen_assistant_prefill",
        )
    elif template_mode == "Apply loom template":
        st.text_input(
            "Loom Filename",
            value="untitled.txt",
            key="multi_gen_loom_filename",
        )

    st.text_area(
        "Prompt",
        key="multi_gen_prompt",
        height=200,
        placeholder="Enter your prompt here...",
    )


def _render_message_builder_tab() -> None:
    """Render the multi-turn message builder tab."""
    if "multi_gen_messages" not in st.session_state:
        st.session_state.multi_gen_messages = []

    messages = st.session_state.multi_gen_messages

    st.text_input(
        "System Prompt",
        key="msg_builder_system",
        placeholder="Optional system prompt...",
    )

    for i, msg in enumerate(messages):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            role = st.selectbox(
                "Role",
                options=["user", "assistant"],
                index=0 if msg["role"] == "user" else 1,
                key=f"msg_role_{i}",
                label_visibility="collapsed",
            )
            if role != msg["role"]:
                messages[i]["role"] = role

        with col2:
            content = st.text_area(
                "Content",
                value=msg["content"],
                key=f"msg_content_{i}",
                height=80,
                label_visibility="collapsed",
            )
            if content != msg["content"]:
                messages[i]["content"] = content

        with col3:
            if st.button("X", key=f"msg_del_{i}"):
                messages.pop(i)
                st.rerun(scope="fragment")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ User", use_container_width=True):
            messages.append({"role": "user", "content": ""})
            st.rerun(scope="fragment")
    with col2:
        if st.button("+ Assistant", use_container_width=True):
            messages.append({"role": "assistant", "content": ""})
            st.rerun(scope="fragment")

    template_override = st.selectbox(
        "Template Override",
        options=[
            "Auto (based on last message)",
            "Force generation prompt",
            "Force continue final message",
        ],
        key="msg_builder_template_override",
    )


def _continue_to_chat(
    result_data: dict, results_data: dict, sample_idx: int = 0
) -> None:
    """Create a new chat from a generation result."""
    from .chat_tab import create_conversation

    mm = result_data["model"]
    response = result_data["results"][sample_idx]
    prompt = results_data.get("prompt", "")
    system_prompt = results_data.get("system_prompt", "")
    messages = results_data.get("messages")

    if messages:
        history = list(messages)
    else:
        history = [{"role": "user", "content": prompt}]

    history.append({"role": "assistant", "content": response})

    create_conversation(
        model_id=mm.model_id,
        name=f"From {mm.config.name}",
        history=history,
        system_prompt=system_prompt,
    )


def _render_result_card(
    idx: int,
    result_data: dict,
    results_data: dict,
    disabled: bool = False,
) -> None:
    """Render a single result card with sample cycling and continue button."""
    mm = result_data["model"]
    key_suffix = "_disabled" if disabled else ""
    samples = result_data["results"]

    with st.expander(f"({idx + 1}) {mm.full_name}", expanded=True):
        render_sample_cycler(
            samples=samples,
            component_id=f"cycler_{idx}{key_suffix}",
            height=300,
        )

        if not disabled:
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(
                    "Continue to Chat",
                    key=f"continue_chat_{idx}",
                    use_container_width=True,
                ):
                    _continue_to_chat(result_data, results_data, sample_idx=0)
                    st.success("Chat created! Go to Chat tab.")


@st.fragment
def render_multi_gen_tab() -> None:
    """Render the Multi-Generation tab."""
    st.markdown("## Multi-Generation")
    st.markdown("Generate text with multiple models side-by-side.")

    active_models = [mm for mm in st.session_state.managed_models.values() if mm.active]

    if not active_models:
        st.warning("No active models. Enable models in the Models tab.")
        return

    _render_import_section()

    text_tab, msg_tab = st.tabs(["Text", "Messages"])

    with text_tab:
        _render_text_input_tab()

    with msg_tab:
        _render_message_builder_tab()

    active_tab = st.session_state.get("multi_gen_active_tab", "Text")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Generate (Text)", type="primary", use_container_width=True):
            st.session_state.multi_gen_active_tab = "Text"
            st.session_state.multi_gen_trigger = True
            st.rerun(scope="fragment")

    with col2:
        messages = st.session_state.get("multi_gen_messages", [])
        if st.button(
            "Generate (Messages)",
            type="primary",
            use_container_width=True,
            disabled=not messages,
        ):
            st.session_state.multi_gen_active_tab = "Messages"
            st.session_state.multi_gen_trigger = True
            st.rerun(scope="fragment")

    if st.session_state.get("multi_gen_trigger"):
        st.session_state.multi_gen_trigger = False
        active_tab = st.session_state.get("multi_gen_active_tab", "Text")
        params = _get_sampling_params()
        inference = st.session_state.inference

        if active_tab == "Text":
            prompt = st.session_state.get("multi_gen_prompt", "")
            if not prompt.strip():
                st.error("Please enter a prompt")
                return

            template_mode = st.session_state.get(
                "multi_gen_template_mode", "Apply chat template"
            )
            system_prompt = st.session_state.get("multi_gen_system_prompt", "")
            assistant_prefill = st.session_state.get("multi_gen_assistant_prefill", "")

            st.markdown("## Generating...")
            output_cols = st.columns(2)
            placeholders = []

            for idx, mm in enumerate(active_models):
                col_idx = idx % 2
                with output_cols[col_idx]:
                    placeholder = st.empty()
                    with placeholder.container():
                        with st.expander(f"({idx + 1}) {mm.full_name}", expanded=True):
                            st.info("Waiting for generation...")
                    placeholders.append(placeholder)

            results = []
            for idx, result_data in enumerate(
                inference.multi_model_sample(
                    models=active_models,
                    prompt_text=prompt,
                    params=params,
                    system_prompt=system_prompt,
                    assistant_prefill=assistant_prefill,
                )
            ):
                results.append(result_data)
                _log_generation(
                    prompt_text=prompt,
                    prompt_tokens=result_data["prompt_tokens"],
                    mm=result_data["model"],
                    outputs=result_data["results"],
                    params=params,
                    system_prompt=system_prompt,
                )
                with placeholders[idx].container():
                    _render_result_card(idx, result_data, {}, disabled=True)

            st.session_state.multi_gen_results = {
                "prompt": prompt,
                "template_mode": template_mode,
                "system_prompt": system_prompt,
                "results": results,
            }
            st.rerun(scope="fragment")

        elif active_tab == "Messages":
            messages = st.session_state.get("multi_gen_messages", [])
            system_prompt = st.session_state.get("msg_builder_system", "")
            template_override = st.session_state.get(
                "msg_builder_template_override", "Auto"
            )

            st.markdown("## Generating from Messages...")
            output_cols = st.columns(2)
            placeholders = []

            for idx, mm in enumerate(active_models):
                col_idx = idx % 2
                with output_cols[col_idx]:
                    placeholder = st.empty()
                    with placeholder.container():
                        with st.expander(f"({idx + 1}) {mm.full_name}", expanded=True):
                            st.info("Waiting for generation...")
                    placeholders.append(placeholder)

            results = []
            for idx, mm in enumerate(active_models):
                tokenizer = inference.get_tokenizer(mm.config.tokenizer_id)

                all_messages = []
                if system_prompt:
                    all_messages.append({"role": "system", "content": system_prompt})
                all_messages.extend(messages)

                if template_override == "Auto (based on last message)":
                    last_role = messages[-1]["role"] if messages else "user"
                    add_gen = last_role == "user"
                    continue_final = last_role == "assistant"
                elif template_override == "Force generation prompt":
                    add_gen = True
                    continue_final = False
                else:
                    add_gen = False
                    continue_final = True

                prompt_tokens = tokenizer.apply_chat_template(
                    all_messages,
                    add_special_tokens=True,
                    add_generation_prompt=add_gen,
                    continue_final_message=continue_final,
                )

                result = inference.sample_from_tokens(mm, prompt_tokens, params)
                result_data = result
                results.append(result_data)

                _log_generation(
                    prompt_text=f"[{len(messages)} messages]",
                    prompt_tokens=prompt_tokens,
                    mm=mm,
                    outputs=result_data["results"],
                    params=params,
                    system_prompt=system_prompt,
                    messages=all_messages,
                )

                with placeholders[idx].container():
                    _render_result_card(idx, result_data, {}, disabled=True)

            st.session_state.multi_gen_results = {
                "messages": messages,
                "system_prompt": system_prompt,
                "results": results,
            }
            st.rerun(scope="fragment")

    if st.session_state.get("multi_gen_results") is not None:
        st.markdown("---")
        results_data = st.session_state.multi_gen_results

        with st.expander("Prompt", expanded=False):
            if "messages" in results_data:
                for msg in results_data["messages"]:
                    st.markdown(f"**{msg['role']}:** {msg['content']}")
            else:
                st.code(
                    results_data.get("prompt", ""), language="text", wrap_lines=True
                )

        st.markdown("## Generated Outputs")
        output_cols = st.columns(2)
        for idx, result_data in enumerate(results_data["results"]):
            col_idx = idx % 2
            with output_cols[col_idx]:
                _render_result_card(idx, result_data, results_data)
