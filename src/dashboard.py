"""
Main Tinker Dashboard.

Streamlit dashboard for sampling from multiple models using tinker.
"""

from pathlib import Path

import streamlit as st

from .inference import TinkerInference
from .dashboard_state import (
    load_models_from_folder,
    load_prompts_from_folder,
    load_loaded_folders,
    load_conversations,
)
from .tabs import (
    render_models_tab,
    render_multi_gen_tab,
    render_chat_tab,
    render_multi_prompt_tab,
)


class TinkerDashboard:
    """Main dashboard class."""

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the dashboard.

        Args:
            cache_dir: Directory for caching configs and state.
                      Defaults to ~/.streamlit_cache/tinker_dashboard
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".streamlit_cache" / "tinker_dashboard"

        self.cache_dir = cache_dir
        self.models_dir = cache_dir / "models"
        self.prompts_dir = cache_dir / "prompts"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self._init_session_state()

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if "inference" not in st.session_state:
            st.session_state.inference = TinkerInference()

        if "cache_dir" not in st.session_state:
            st.session_state.cache_dir = self.cache_dir
        if "models_dir" not in st.session_state:
            st.session_state.models_dir = self.models_dir
        if "prompts_dir" not in st.session_state:
            st.session_state.prompts_dir = self.prompts_dir

        if "managed_models" not in st.session_state:
            st.session_state.managed_models = {}
        if "managed_prompts" not in st.session_state:
            st.session_state.managed_prompts = {}

        if "loaded_model_folders" not in st.session_state:
            model_folders, prompt_folders = load_loaded_folders(
                self.cache_dir / "loaded_folders.yaml",
                self.models_dir,
                self.prompts_dir,
            )
            st.session_state.loaded_model_folders = model_folders
            st.session_state.loaded_prompt_folders = prompt_folders

            for folder in model_folders:
                loaded = load_models_from_folder(self.models_dir, folder)
                st.session_state.managed_models.update(loaded)
            for folder in prompt_folders:
                loaded = load_prompts_from_folder(self.prompts_dir, folder)
                st.session_state.managed_prompts.update(loaded)

        if "conversations" not in st.session_state:
            conv_dir = self.cache_dir / "conversations"
            st.session_state.conversations = load_conversations(conv_dir)
        if "conversation_counter" not in st.session_state:
            existing_ids = [
                int(cid.split("_")[1])
                for cid in st.session_state.conversations.keys()
                if cid.startswith("conv_") and cid.split("_")[1].isdigit()
            ]
            st.session_state.conversation_counter = max(existing_ids, default=0) + 1

        if "sampling_params" not in st.session_state:
            st.session_state.sampling_params = {
                "temperature": 1.0,
                "top_p": 0.9,
                "max_tokens": 2048,
                "n": 4,
                "seed": 42,
                "skip_special_tokens": False,
            }

        if "multi_gen_results" not in st.session_state:
            st.session_state.multi_gen_results = None
        if "multi_prompt_results" not in st.session_state:
            st.session_state.multi_prompt_results = None

    def _render_sidebar(self) -> None:
        """Render sidebar with sampling parameters."""
        with st.sidebar.expander("Sampling Parameters", expanded=True):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.sampling_params["temperature"],
                step=0.1,
            )
            top_p = st.slider(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.sampling_params["top_p"],
                step=0.05,
            )
            max_tokens = st.slider(
                "Max Tokens",
                min_value=10,
                max_value=4096,
                value=st.session_state.sampling_params["max_tokens"],
                step=10,
            )
            n = st.slider(
                "Num Samples",
                min_value=1,
                max_value=16,
                value=st.session_state.sampling_params["n"],
                step=1,
            )
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=st.session_state.sampling_params["seed"],
                step=1,
            )
            skip_special = st.checkbox(
                "Skip Special Tokens",
                value=st.session_state.sampling_params["skip_special_tokens"],
            )

            st.session_state.sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "seed": seed,
                "skip_special_tokens": skip_special,
            }

        with st.sidebar.expander("Quick Actions", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Enable All", use_container_width=True):
                    for mm in st.session_state.managed_models.values():
                        mm.active = True
                    st.rerun()
            with col2:
                if st.button("Disable All", use_container_width=True):
                    for mm in st.session_state.managed_models.values():
                        mm.active = False
                    st.rerun()

            active_count = sum(
                1 for mm in st.session_state.managed_models.values() if mm.active
            )
            total_count = len(st.session_state.managed_models)

    def display(self) -> None:
        """Main entry point for the dashboard."""
        st.set_page_config(
            page_title="Tinker Dashboard",
            page_icon="",
            layout="wide",
        )

        st.title("Tinker Dashboard")

        self._render_sidebar()

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Models",
                "Multi-Generation",
                "Chat",
                "Multi-Prompt",
            ]
        )

        with tab1:
            render_models_tab()
        with tab2:
            render_multi_gen_tab()
        with tab3:
            render_chat_tab()
        with tab4:
            render_multi_prompt_tab()
