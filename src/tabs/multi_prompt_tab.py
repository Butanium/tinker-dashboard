"""
Multi-prompt tab.

Run multiple prompts across selected models for batch generation.
"""

from copy import deepcopy

import streamlit as st

from ..dashboard_state import (
    ManagedPrompt,
    save_prompts_to_folder,
    load_prompts_from_folder,
    unload_folder_prompts,
    get_unique_name,
)
from ..folder_manager_ui import FolderManagerUI, FolderManagerConfig
from ..inference import SamplingParams
from .multi_gen_tab import render_sample_cycler


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


def _create_new_prompt(folder: str | None) -> ManagedPrompt:
    """Create a new prompt."""
    return ManagedPrompt(
        prompt_mode="text",
        active=True,
        expanded=True,
        folder=folder,
    )


def _save_prompts() -> None:
    """Save all prompts to their folders."""
    folders = {mp.folder for mp in st.session_state.managed_prompts.values()}
    for folder in folders:
        save_prompts_to_folder(
            st.session_state.managed_prompts,
            st.session_state.prompts_dir,
            folder,
        )


def _save_loaded_folders() -> None:
    """Save which folders are loaded."""
    from ..dashboard_state import save_loaded_folders

    save_loaded_folders(
        st.session_state.cache_dir / "loaded_folders.yaml",
        st.session_state.loaded_model_folders,
        st.session_state.loaded_prompt_folders,
    )


def _render_text_prompt_editor(prompt_id: str, mp: ManagedPrompt) -> None:
    """Render text prompt editor."""
    template_mode = st.selectbox(
        "Template Mode",
        options=["Apply chat template", "No template", "Apply loom template"],
        index=(
            ["Apply chat template", "No template", "Apply loom template"].index(
                mp.template_mode
            )
            if mp.template_mode
            in ["Apply chat template", "No template", "Apply loom template"]
            else 0
        ),
        key=f"prompt_template_{prompt_id}",
    )
    if template_mode != mp.template_mode:
        mp.template_mode = template_mode
        _save_prompts()

    if template_mode == "Apply chat template":
        system_prompt = st.text_area(
            "System Prompt (optional)",
            value=mp.system_prompt,
            key=f"prompt_sys_{prompt_id}",
            height=68,
        )
        if system_prompt != mp.system_prompt:
            mp.system_prompt = system_prompt
            _save_prompts()

    content = st.text_area(
        "Prompt",
        value=mp.content,
        key=f"prompt_content_{prompt_id}",
        height=150,
    )
    if content != mp.content:
        mp.content = content
        _save_prompts()


def _render_messages_prompt_editor(prompt_id: str, mp: ManagedPrompt) -> None:
    """Render messages (multi-turn) prompt editor."""
    system_prompt = st.text_input(
        "System Prompt",
        value=mp.system_prompt,
        key=f"prompt_sys_msg_{prompt_id}",
        placeholder="Optional system prompt...",
    )
    if system_prompt != mp.system_prompt:
        mp.system_prompt = system_prompt
        _save_prompts()

    for i, msg in enumerate(mp.messages):
        col1, col2, col3 = st.columns([1, 8, 1])
        with col1:
            role = st.selectbox(
                "Role",
                options=["user", "assistant"],
                index=0 if msg["role"] == "user" else 1,
                key=f"prompt_msg_role_{prompt_id}_{i}",
                label_visibility="collapsed",
            )
            if role != msg["role"]:
                mp.messages[i]["role"] = role
                _save_prompts()

        with col2:
            content = st.text_area(
                "Content",
                value=msg["content"],
                key=f"prompt_msg_content_{prompt_id}_{i}",
                height=80,
                label_visibility="collapsed",
            )
            if content != msg["content"]:
                mp.messages[i]["content"] = content
                _save_prompts()

        with col3:
            if st.button("X", key=f"prompt_msg_del_{prompt_id}_{i}"):
                mp.messages.pop(i)
                _save_prompts()
                st.rerun(scope="fragment")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "+ User", key=f"prompt_add_user_{prompt_id}", use_container_width=True
        ):
            mp.messages.append({"role": "user", "content": ""})
            _save_prompts()
            st.rerun(scope="fragment")
    with col2:
        if st.button(
            "+ Assistant", key=f"prompt_add_asst_{prompt_id}", use_container_width=True
        ):
            mp.messages.append({"role": "assistant", "content": ""})
            _save_prompts()
            st.rerun(scope="fragment")


def _render_prompt_editor(prompt_id: str, mp: ManagedPrompt) -> None:
    """Render the editor for a single prompt."""
    icon = "+" if mp.active else "-"
    display_name = mp.get_display_name() or "New Prompt"

    with st.expander(f"{icon} {display_name}", expanded=mp.expanded):
        name = st.text_input(
            "Name (optional)",
            value=mp.name,
            key=f"prompt_name_{prompt_id}",
            placeholder="Auto-generated from content",
        )
        if name != mp.name:
            mp.name = name
            _save_prompts()

        prompt_mode = st.selectbox(
            "Prompt Mode",
            options=["text", "messages"],
            index=0 if mp.prompt_mode == "text" else 1,
            key=f"prompt_mode_{prompt_id}",
        )
        if prompt_mode != mp.prompt_mode:
            mp.prompt_mode = prompt_mode
            _save_prompts()

        if mp.prompt_mode == "text":
            _render_text_prompt_editor(prompt_id, mp)
        else:
            _render_messages_prompt_editor(prompt_id, mp)

        active = st.checkbox(
            "Active",
            value=mp.active,
            key=f"prompt_active_{prompt_id}",
        )
        if active != mp.active:
            mp.active = active
            _save_prompts()


def _render_prompt_actions(prompt_id: str, mp: ManagedPrompt) -> None:
    """Render duplicate/delete buttons for a prompt."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Dup", key=f"dup_prompt_{prompt_id}", help="Duplicate"):
            new_mp = deepcopy(mp)
            new_mp.name = get_unique_name(
                f"{mp.name or 'Prompt'} copy",
                {p.name for p in st.session_state.managed_prompts.values()},
            )
            new_mp.prompt_id = None
            new_mp = ManagedPrompt(
                name=new_mp.name,
                prompt_mode=mp.prompt_mode,
                content=mp.content,
                messages=list(mp.messages),
                system_prompt=mp.system_prompt,
                template_mode=mp.template_mode,
                folder=mp.folder,
                active=mp.active,
                expanded=True,
            )
            st.session_state.managed_prompts[new_mp.prompt_id] = new_mp
            _save_prompts()
            st.rerun(scope="fragment")

    with col2:
        if st.button("Del", key=f"del_prompt_{prompt_id}", help="Delete"):
            del st.session_state.managed_prompts[prompt_id]
            _save_prompts()
            st.rerun(scope="fragment")


def _tokenize_prompt(mp: ManagedPrompt, tokenizer) -> list[int]:
    """Tokenize a prompt based on its mode."""
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
                    {
                        "role": "user",
                        "content": f"<cmd>cat untitled.txt</cmd>",
                    },
                    {"role": "assistant", "content": mp.content},
                ],
                continue_final_message=True,
            )
    else:
        all_messages = []
        if mp.system_prompt:
            all_messages.append({"role": "system", "content": mp.system_prompt})
        all_messages.extend(mp.messages)

        if mp.messages:
            last_role = mp.messages[-1]["role"]
            add_gen = last_role == "user"
            continue_final = last_role == "assistant"
        else:
            add_gen = True
            continue_final = False

        return tokenizer.apply_chat_template(
            all_messages,
            add_special_tokens=True,
            add_generation_prompt=add_gen,
            continue_final_message=continue_final,
        )


def _run_multi_prompt_generation(
    active_prompts: list[ManagedPrompt],
    selected_models: list,
) -> None:
    """Run generation across all prompts and models."""
    params = _get_sampling_params()
    inference = st.session_state.inference

    results = []

    progress = st.progress(0, text="Generating...")
    total = len(active_prompts) * len(selected_models)
    current = 0

    for mp in active_prompts:
        prompt_results = {"prompt": mp, "models": []}

        for mm in selected_models:
            tokenizer = inference.get_tokenizer(mm.config.tokenizer_id)
            prompt_tokens = _tokenize_prompt(mp, tokenizer)

            result = inference.sample_from_tokens(mm, prompt_tokens, params)
            prompt_results["models"].append(result)

            current += 1
            progress.progress(
                current / total, text=f"Generating... ({current}/{total})"
            )

        results.append(prompt_results)

    progress.empty()
    st.session_state.multi_prompt_results = results


def _render_results() -> None:
    """Render the multi-prompt results."""
    results = st.session_state.get("multi_prompt_results")
    if results is None:
        st.info("No results yet. Select prompts and models, then click Run.")
        return

    for prompt_result in results:
        mp = prompt_result["prompt"]
        st.markdown(f"### {mp.get_display_name()}")

        with st.expander("Prompt", expanded=False):
            if mp.prompt_mode == "messages":
                for msg in mp.messages:
                    st.markdown(f"**{msg['role']}:** {msg['content']}")
            else:
                st.code(mp.content, language="text", wrap_lines=True)

        cols = st.columns(min(len(prompt_result["models"]), 3))
        for idx, model_result in enumerate(prompt_result["models"]):
            mm = model_result["model"]
            col_idx = idx % len(cols)
            with cols[col_idx]:
                with st.expander(mm.config.name, expanded=True):
                    render_sample_cycler(
                        samples=model_result["results"],
                        component_id=f"mp_cycler_{mp.prompt_id}_{mm.model_id}",
                        height=200,
                    )

        st.markdown("---")


@st.fragment
def render_multi_prompt_tab() -> None:
    """Render the Multi-Prompt tab."""
    active_prompts = [
        mp for mp in st.session_state.managed_prompts.values() if mp.active
    ]
    active_models = [mm for mm in st.session_state.managed_models.values() if mm.active]

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if active_models:
            model_ids = [mm.model_id for mm in active_models]
            model_names = {mm.model_id: mm.config.name for mm in active_models}
            selected_ids = st.multiselect(
                "Models to run",
                options=model_ids,
                default=model_ids[:3],
                format_func=lambda x: model_names.get(x, x),
                key="multi_prompt_models",
            )
            selected_models = [
                mm for mm in active_models if mm.model_id in selected_ids
            ]
        else:
            st.warning("No active models")
            selected_models = []

    with col2:
        st.write(
            f"**{len(active_prompts)} prompt(s), {len(selected_models)} model(s)**"
        )

    with col3:
        if st.button(
            f"Run ({len(active_prompts)})",
            use_container_width=True,
            disabled=not active_prompts or not selected_models,
        ):
            _run_multi_prompt_generation(active_prompts, selected_models)
            st.rerun(scope="fragment")

    prompts_tab, results_tab = st.tabs(["Prompts", "Results"])

    with prompts_tab:
        folder_manager = FolderManagerUI(
            FolderManagerConfig(
                base_dir=st.session_state.prompts_dir,
                loaded_folders_key="loaded_prompt_folders",
                items_key="managed_prompts",
                item_type_label="prompt",
                widget_key_prefix="prompt_folder",
                load_from_folder=lambda base, folder: load_prompts_from_folder(
                    base, folder
                ),
                save_to_folder=save_prompts_to_folder,
                unload_folder=unload_folder_prompts,
                create_new_item=_create_new_prompt,
                get_item_folder=lambda mp: mp.folder,
                save_loaded_folders=_save_loaded_folders,
                save_items=_save_prompts,
                rerun_scope="fragment",
            )
        )

        folder_manager.render_folder_loader()
        st.markdown("---")
        folder_manager.render_all_folders(
            render_item=_render_prompt_editor,
            render_item_actions=_render_prompt_actions,
        )

    with results_tab:
        _render_results()
