"""Dashboard tab components."""

from .models_tab import render_models_tab
from .multi_gen_tab import render_multi_gen_tab
from .chat_tab import render_chat_tab
from .multi_prompt_tab import render_multi_prompt_tab

__all__ = [
    "render_models_tab",
    "render_multi_gen_tab",
    "render_chat_tab",
    "render_multi_prompt_tab",
]
