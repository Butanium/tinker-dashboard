"""
Models configuration tab.

Allows users to add, edit, and manage model configurations.
"""

from copy import deepcopy

import streamlit as st

from ..model_config import ModelConfig, ManagedModel
from ..folder_manager_ui import FolderManagerUI, FolderManagerConfig
from ..dashboard_state import (
    save_models_to_folder,
    load_models_from_folder,
    unload_folder_models,
    get_unique_name,
)


def _create_new_model(folder: str | None) -> ManagedModel:
    """Create a new model configuration."""
    existing_names = {mm.config.name for mm in st.session_state.managed_models.values()}
    unique_name = get_unique_name("New Model", existing_names)

    config = ModelConfig(
        name=unique_name,
        tokenizer_id="meta-llama/Llama-3.3-70B-Instruct",
        sampler_path="tinker://",
        description="",
    )
    return ManagedModel.from_config(config, active=True, expanded=True, folder=folder)


def _save_models() -> None:
    """Save all models to their folders."""
    folders = {mm.folder for mm in st.session_state.managed_models.values()}
    for folder in folders:
        save_models_to_folder(
            st.session_state.managed_models,
            st.session_state.models_dir,
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


def _render_model_editor(model_id: str, mm: ManagedModel) -> None:
    """Render the editor for a single model configuration."""
    icon = "+" if mm.active else "-"
    display_name = mm.config.name

    with st.expander(f"{icon} {display_name}", expanded=mm.expanded):
        col1, col2 = st.columns([1, 1])

        with col1:
            name = st.text_input(
                "Name",
                value=mm.config.name,
                key=f"model_name_{model_id}",
            )
            if name != mm.config.name:
                mm.config.name = name
                _save_models()

            tokenizer_id = st.text_input(
                "Tokenizer ID",
                value=mm.config.tokenizer_id,
                key=f"tokenizer_{model_id}",
                help="HuggingFace model ID for tokenizer",
            )
            if tokenizer_id != mm.config.tokenizer_id:
                mm.config.tokenizer_id = tokenizer_id
                _save_models()

        with col2:
            sampler_path = st.text_input(
                "Sampler Path",
                value=mm.config.sampler_path,
                key=f"sampler_{model_id}",
                help="tinker:// URI to sampler weights",
            )
            if sampler_path != mm.config.sampler_path:
                mm.config.sampler_path = sampler_path
                _save_models()

        description = st.text_area(
            "Description",
            value=mm.config.description,
            key=f"desc_{model_id}",
            height=68,
        )
        if description != mm.config.description:
            mm.config.description = description
            _save_models()

        active = st.checkbox(
            "Active",
            value=mm.active,
            key=f"active_{model_id}",
            help="Include this model in generation",
        )
        if active != mm.active:
            mm.active = active
            _save_models()


def _render_model_actions(model_id: str, mm: ManagedModel) -> None:
    """Render duplicate/delete buttons for a model."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Dup", key=f"dup_{model_id}", help="Duplicate"):
            new_config = deepcopy(mm.config)
            existing = {m.config.name for m in st.session_state.managed_models.values()}
            new_config.name = get_unique_name(f"{mm.config.name} copy", existing)
            new_mm = ManagedModel.from_config(
                new_config, active=mm.active, expanded=True, folder=mm.folder
            )
            st.session_state.managed_models[new_mm.model_id] = new_mm
            _save_models()
            st.rerun(scope="fragment")

    with col2:
        if st.button("Del", key=f"del_{model_id}", help="Delete"):
            del st.session_state.managed_models[model_id]
            _save_models()
            st.rerun(scope="fragment")


@st.fragment
def render_models_tab() -> None:
    """Render the Models configuration tab."""
    st.markdown("## Models")
    st.markdown("Configure model endpoints for generation.")

    folder_manager = FolderManagerUI(
        FolderManagerConfig(
            base_dir=st.session_state.models_dir,
            loaded_folders_key="loaded_model_folders",
            items_key="managed_models",
            item_type_label="model",
            widget_key_prefix="model_folder",
            load_from_folder=lambda base, folder: load_models_from_folder(
                base,
                folder,
                {mm.full_name for mm in st.session_state.managed_models.values()},
            ),
            save_to_folder=save_models_to_folder,
            unload_folder=unload_folder_models,
            create_new_item=_create_new_model,
            get_item_folder=lambda mm: mm.folder,
            save_loaded_folders=_save_loaded_folders,
            save_items=_save_models,
            rerun_scope="fragment",
        )
    )

    folder_manager.render_folder_loader()
    st.markdown("---")
    folder_manager.render_all_folders(
        render_item=_render_model_editor,
        render_item_actions=_render_model_actions,
    )
