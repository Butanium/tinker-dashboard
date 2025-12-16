"""
State persistence for the tinker dashboard.

Handles saving/loading models and prompts to disk.
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import re

import yaml

from .model_config import ModelConfig, ManagedModel


def sanitize_name(name: str) -> str:
    """Convert name to filesystem-safe string."""
    safe = re.sub(r"[^\w\s-]", "", name)
    safe = re.sub(r"[\s]+", "_", safe)
    return safe.lower().strip("_")[:50]


def get_unique_name(base_name: str, existing_names: set[str]) -> str:
    """Get unique name by appending number if needed."""
    if base_name not in existing_names:
        return base_name
    counter = 2
    while f"{base_name} ({counter})" in existing_names:
        counter += 1
    return f"{base_name} ({counter})"


@dataclass
class ManagedPrompt:
    """A managed prompt for batch generation."""

    name: str = ""
    content: str = ""
    messages: list[dict] = field(default_factory=list)
    use_chat_format: bool = False
    folder: str | None = None
    prompt_id: str = field(default_factory=lambda: str(uuid4()))
    active: bool = True
    expanded: bool = False

    def get_display_name(self) -> str:
        """Get name or truncated content."""
        if self.name:
            return self.name
        text = self.content or (self.messages[0]["content"] if self.messages else "")
        return text[:40] + "..." if len(text) > 40 else text

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "content": self.content,
            "messages": self.messages,
            "use_chat_format": self.use_chat_format,
            "folder": self.folder,
            "prompt_id": self.prompt_id,
            "active": self.active,
            "expanded": self.expanded,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManagedPrompt":
        """Deserialize from dict."""
        return cls(
            name=data.get("name", ""),
            content=data.get("content", ""),
            messages=data.get("messages", []),
            use_chat_format=data.get("use_chat_format", False),
            folder=data.get("folder"),
            prompt_id=data.get("prompt_id", str(uuid4())),
            active=data.get("active", True),
            expanded=data.get("expanded", False),
        )


def _get_folder_path(base_dir: Path, folder: str | None) -> Path:
    """Get the path for a folder (None = root)."""
    if folder is None:
        return base_dir
    return base_dir / folder


def _save_ui_state(path: Path, items: dict, get_folder: callable) -> None:
    """Save UI state (active, expanded, ui_order) for items in a folder."""
    folder = None
    for item in items.values():
        folder = get_folder(item)
        break

    folder_items = {k: v for k, v in items.items() if get_folder(v) == folder}
    ui_state = {}
    for item_id, item in folder_items.items():
        name = item.config.name if hasattr(item, "config") else item.name
        ui_state[name] = {
            "active": item.active,
            "expanded": item.expanded,
            "ui_order": getattr(item, "ui_order", 0),
        }

    with open(path, "w") as f:
        yaml.safe_dump(ui_state, f)


def _load_ui_state(path: Path) -> dict:
    """Load UI state from file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Model persistence

def save_models_to_folder(
    models: dict[str, ManagedModel], base_dir: Path, folder: str | None
) -> None:
    """Save models to a folder, with _ui_state.yaml for UI state."""
    folder_path = _get_folder_path(base_dir, folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    folder_models = {k: v for k, v in models.items() if v.folder == folder}

    for model_id, mm in folder_models.items():
        filename = sanitize_name(mm.config.name) + ".yaml"
        filepath = folder_path / filename
        with open(filepath, "w") as f:
            yaml.safe_dump(mm.config.to_dict(), f)

    ui_state = {}
    for mm in folder_models.values():
        ui_state[mm.config.name] = {
            "active": mm.active,
            "expanded": mm.expanded,
            "ui_order": mm.ui_order,
        }
    with open(folder_path / "_ui_state.yaml", "w") as f:
        yaml.safe_dump(ui_state, f)


def load_models_from_folder(
    base_dir: Path, folder: str | None, existing_names: set[str] | None = None
) -> dict[str, ManagedModel]:
    """Load models from a folder."""
    folder_path = _get_folder_path(base_dir, folder)
    if not folder_path.exists():
        return {}

    existing_names = existing_names or set()
    ui_state = _load_ui_state(folder_path / "_ui_state.yaml")

    result = {}
    for filepath in folder_path.glob("*.yaml"):
        if filepath.name.startswith("_"):
            continue

        with open(filepath) as f:
            data = yaml.safe_load(f)

        config = ModelConfig.from_dict(data)
        state = ui_state.get(config.name, {})

        mm = ManagedModel(
            config=config,
            folder=folder,
            active=state.get("active", True),
            expanded=state.get("expanded", False),
            ui_order=state.get("ui_order", 0),
        )
        result[mm.model_id] = mm

    return result


def unload_folder_models(
    models: dict[str, ManagedModel], folder: str | None
) -> dict[str, ManagedModel]:
    """Remove models from the given folder, return remaining models."""
    return {k: v for k, v in models.items() if v.folder != folder}


# Prompt persistence

def save_prompts_to_folder(
    prompts: dict[str, ManagedPrompt], base_dir: Path, folder: str | None
) -> None:
    """Save prompts to a folder."""
    folder_path = _get_folder_path(base_dir, folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    folder_prompts = {k: v for k, v in prompts.items() if v.folder == folder}

    for prompt_id, mp in folder_prompts.items():
        name = mp.name or f"prompt_{prompt_id[:8]}"
        filename = sanitize_name(name) + ".yaml"
        filepath = folder_path / filename
        data = {
            "name": mp.name,
            "content": mp.content,
            "messages": mp.messages,
            "use_chat_format": mp.use_chat_format,
        }
        with open(filepath, "w") as f:
            yaml.safe_dump(data, f)

    ui_state = {}
    for mp in folder_prompts.values():
        name = mp.name or mp.prompt_id
        ui_state[name] = {"active": mp.active, "expanded": mp.expanded}
    with open(folder_path / "_ui_state.yaml", "w") as f:
        yaml.safe_dump(ui_state, f)


def load_prompts_from_folder(
    base_dir: Path, folder: str | None
) -> dict[str, ManagedPrompt]:
    """Load prompts from a folder."""
    folder_path = _get_folder_path(base_dir, folder)
    if not folder_path.exists():
        return {}

    ui_state = _load_ui_state(folder_path / "_ui_state.yaml")

    result = {}
    for filepath in folder_path.glob("*.yaml"):
        if filepath.name.startswith("_"):
            continue

        with open(filepath) as f:
            data = yaml.safe_load(f)

        name = data.get("name", filepath.stem)
        state = ui_state.get(name, {})

        mp = ManagedPrompt(
            name=name,
            content=data.get("content", ""),
            messages=data.get("messages", []),
            use_chat_format=data.get("use_chat_format", False),
            folder=folder,
            active=state.get("active", True),
            expanded=state.get("expanded", False),
        )
        result[mp.prompt_id] = mp

    return result


def unload_folder_prompts(
    prompts: dict[str, ManagedPrompt], folder: str | None
) -> dict[str, ManagedPrompt]:
    """Remove prompts from the given folder, return remaining prompts."""
    return {k: v for k, v in prompts.items() if v.folder != folder}


# Loaded folders state

def save_loaded_folders(
    path: Path, model_folders: set[str | None], prompt_folders: set[str | None]
) -> None:
    """Save which folders are currently loaded."""
    data = {
        "model_folders": [f for f in model_folders],
        "prompt_folders": [f for f in prompt_folders],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_loaded_folders(
    path: Path, models_dir: Path, prompts_dir: Path
) -> tuple[set[str | None], set[str | None]]:
    """Load which folders were previously loaded and reload their contents."""
    if not path.exists():
        return set(), set()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    model_folders = set(data.get("model_folders", []))
    prompt_folders = set(data.get("prompt_folders", []))

    return model_folders, prompt_folders
