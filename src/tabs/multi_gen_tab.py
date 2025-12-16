"""
Multi-generation tab.

Run multiple models on the same prompt and compare outputs side-by-side.
"""

import html
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ..inference import SamplingParams


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


def _render_text_input() -> tuple[str, str, str, str]:
    """Render the text input section and return prompt details."""
    template_mode = st.selectbox(
        "Template Mode",
        options=["Apply chat template", "No template", "Apply loom template"],
        key="multi_gen_template_mode",
    )

    system_prompt = ""
    assistant_prefill = ""
    loom_filename = "untitled.txt"

    if template_mode == "Apply chat template":
        system_prompt = st.text_area(
            "System Prompt (optional)",
            key="multi_gen_system_prompt",
            height=68,
        )
        assistant_prefill = st.text_input(
            "Assistant Prefill (optional)",
            key="multi_gen_assistant_prefill",
        )
    elif template_mode == "Apply loom template":
        loom_filename = st.text_input(
            "Loom Filename",
            value="untitled.txt",
            key="multi_gen_loom_filename",
        )

    prompt = st.text_area(
        "Prompt",
        key="multi_gen_prompt",
        height=200,
        placeholder="Enter your prompt here...",
    )

    return prompt, template_mode, system_prompt, assistant_prefill


def _render_result_card(idx: int, result_data: dict, disabled: bool = False) -> None:
    """Render a single result card with sample cycling."""
    mm = result_data["model"]
    key_suffix = "_disabled" if disabled else ""

    with st.expander(f"({idx + 1}) {mm.full_name}", expanded=True):
        render_sample_cycler(
            samples=result_data["results"],
            component_id=f"cycler_{idx}{key_suffix}",
            height=300,
        )


@st.fragment
def render_multi_gen_tab() -> None:
    """Render the Multi-Generation tab."""
    st.markdown("## Multi-Generation")
    st.markdown("Generate text with multiple models side-by-side.")

    active_models = [
        mm for mm in st.session_state.managed_models.values() if mm.active
    ]

    if not active_models:
        st.warning("No active models. Enable models in the Models tab.")
        return

    st.info(f"**{len(active_models)} active model(s)**")

    prompt, template_mode, system_prompt, assistant_prefill = _render_text_input()

    col1, col2 = st.columns([1, 3])
    with col1:
        generate_clicked = st.button(
            "Generate",
            type="primary",
            use_container_width=True,
            disabled=not prompt.strip(),
        )

    if generate_clicked and prompt.strip():
        params = _get_sampling_params()
        inference = st.session_state.inference

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
            with placeholders[idx].container():
                _render_result_card(idx, result_data, disabled=True)

        st.session_state.multi_gen_results = {
            "prompt": prompt,
            "template_mode": template_mode,
            "system_prompt": system_prompt,
            "results": results,
        }
        st.rerun(scope="fragment")

    if st.session_state.get("multi_gen_results") is not None:
        st.markdown("---")
        results_data = st.session_state.multi_gen_results

        with st.expander("Prompt", expanded=False):
            st.code(results_data["prompt"], language="text", wrap_lines=True)

        st.markdown("## Generated Outputs")
        output_cols = st.columns(2)
        for idx, result_data in enumerate(results_data["results"]):
            col_idx = idx % 2
            with output_cols[col_idx]:
                _render_result_card(idx, result_data)
