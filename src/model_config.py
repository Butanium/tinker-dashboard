"""
Model configuration dataclasses for the tinker dashboard.
"""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ModelConfig:
    """Configuration for a model endpoint."""

    name: str
    base_model: str  # Base model name (also used as tokenizer)
    sampler_path: str = ""  # tinker:// URI to sampler weights (empty for base model)
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for YAML storage."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "sampler_path": self.sampler_path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Deserialize from dict."""
        # Support legacy tokenizer_id field
        base_model = data.get("base_model") or data.get("tokenizer_id", "")
        return cls(
            name=data["name"],
            base_model=base_model,
            sampler_path=data.get("sampler_path", ""),
            description=data.get("description", ""),
        )


@dataclass
class ManagedModel:
    """UI wrapper around ModelConfig with dashboard state."""

    config: ModelConfig
    folder: str | None = None
    model_id: str = field(default_factory=lambda: str(uuid4()))
    active: bool = True
    expanded: bool = False
    ui_order: int = 0

    @property
    def full_name(self) -> str:
        """Get display name with folder prefix."""
        if self.folder:
            return f"{self.folder}/{self.config.name}"
        return self.config.name

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            "config": self.config.to_dict(),
            "folder": self.folder,
            "model_id": self.model_id,
            "active": self.active,
            "expanded": self.expanded,
            "ui_order": self.ui_order,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManagedModel":
        """Deserialize from dict."""
        return cls(
            config=ModelConfig.from_dict(data["config"]),
            folder=data.get("folder"),
            model_id=data.get("model_id", str(uuid4())),
            active=data.get("active", True),
            expanded=data.get("expanded", False),
            ui_order=data.get("ui_order", 0),
        )

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        active: bool = True,
        expanded: bool = False,
        folder: str | None = None,
    ) -> "ManagedModel":
        """Create ManagedModel from a ModelConfig."""
        return cls(
            config=config,
            folder=folder,
            active=active,
            expanded=expanded,
        )
