# Tinker Dashboard

A Streamlit-based web application for sampling from and comparing multiple language models. Tinker Dashboard provides a unified interface for multi-model inference, allowing you to evaluate and compare model outputs across different sampling parameters.

## Features

- **Multi-model sampling**: Generate outputs from multiple models simultaneously with the same prompt
- **Model management**: Configure, organize, and manage model endpoints with custom tokenizers and sampler paths
- **Prompt management**: Create and organize prompts in folders for batch evaluation
- **Chat interface**: Interactive chat with selectable models
- **Multi-prompt evaluation**: Test multiple prompts across multiple models to compare behaviors
- **Configurable sampling**: Adjust temperature, top-p, max tokens, number of samples, and seed
- **Persistent state**: All configurations, conversations, and generation logs are saved locally

## Installation

### Requirements

- Python 3.13+
- `uv` package manager

### Setup

1. Clone or download the project
2. Install dependencies:
   ```bash
   uv sync
   ```

3. Configure your Tinker service credentials:
   - Create or update `~/.env` with your tinker service configuration
   - Or set `TINKER_*` environment variables as needed

## Running the Dashboard

```bash
uv run streamlit run app.py
```

The application will launch at `http://localhost:8501` by default.

## Project Structure

```
tinker-dashboard/
├── app.py                     # Main entry point
├── example.py                 # Example usage
├── src/
│   ├── dashboard.py           # Main TinkerDashboard class
│   ├── dashboard_state.py     # State persistence (models, prompts, conversations)
│   ├── model_config.py        # Model configuration dataclasses
│   ├── inference.py           # TinkerInference wrapper for sampling
│   ├── tokenizers.py          # Tokenizer caching utility
│   ├── folder_manager_ui.py   # Generic folder management UI component
│   └── tabs/
│       ├── models_tab.py      # Model configuration UI
│       ├── multi_gen_tab.py   # Multi-model generation tab
│       ├── chat_tab.py        # Chat interface tab
│       └── multi_prompt_tab.py # Multi-prompt evaluation tab
└── pyproject.toml             # Project configuration
```

## Usage

### 1. Models Tab

Configure and manage model endpoints:
- **Name**: Identifier for the model
- **Tokenizer ID**: HuggingFace model ID for the tokenizer
- **Sampler Path**: tinker:// URI to the sampler weights
- **Description**: Optional notes about the model
- **Active**: Toggle whether the model is used in generation

### 2. Multi-Generation Tab

Generate samples from multiple models with a single prompt:
1. Select one or more models from the sidebar
2. Enter your prompt
3. Adjust sampling parameters (temperature, top-p, etc.)
4. Generate and compare outputs

### 3. Chat Tab

Interactive chat with selectable models:
- Maintains conversation history
- Switch between models mid-conversation
- All interactions are persisted

### 4. Multi-Prompt Tab

Evaluate multiple prompts across multiple models:
- Test different prompts with the same models
- Compare behaviors across prompt variations
- Useful for prompt optimization and model comparison

## Configuration

### Sampling Parameters

Adjust in the sidebar for all generation tabs:
- **Temperature** (0.0-2.0): Controls randomness
- **Top-p** (0.0-1.0): Nucleus sampling threshold
- **Max Tokens** (10-4096): Maximum output length
- **Samples** (1-16): Number of samples per model
- **Seed**: For reproducible generations
- **Skip Special Tokens**: Exclude special tokens from output

### Data Storage

All data is persisted locally:
- **Models & Prompts**: `~/.streamlit_cache/tinker_dashboard/models/` and `prompts/`
- **Conversations**: `~/.streamlit_cache/tinker_dashboard/conversations/`
- **Generation Logs**: Available in the dashboard for audit/analysis

## Dependencies

- **streamlit** - Web UI framework
- **tinker** - Language model sampling library
- **transformers** - HuggingFace transformers for tokenizers
- **pyyaml** - YAML configuration parsing
- **python-dotenv** - Environment variable management

See `pyproject.toml` for version constraints.

## Development

To add a new package:
```bash
uv add <package-name>
```

Never edit `pyproject.toml` directly; use `uv add` instead.

## License

[Specify your license here]
