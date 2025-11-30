# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is dedicated to training LLaMA models to improve code quality advice. The training data consists of:

- **Documentation**: High-quality technical documentation and code references
- **Best practices**: Established patterns and conventions for writing good code
- **Example code**: Both good examples (to reinforce) and bad examples (to teach what to avoid)

The goal is to fine-tune LLaMA models to provide better recommendations, identify code smells, suggest improvements, and understand the nuances between good and bad coding practices.

## Directory Structure

- `examples/` - Good and bad code examples for training data
  - Organize by language, pattern, or anti-pattern
  - Include both positive examples (what to do) and negative examples (what to avoid)
- `docs/` - Documentation corpus for training
  - Style guides, API documentation, architectural guides
- `best-practices/` - Curated best practices and coding standards
  - Language-specific conventions
  - Design patterns and principles

## Training Data Organization

When adding training data:

1. **Label examples clearly**: Mark code as "good" or "bad" with explanations of why
2. **Include context**: Provide documentation explaining the reasoning behind best practices
3. **Pair contrasts**: When possible, show both good and bad implementations of the same concept
4. **Add metadata**: Consider including language, complexity level, and specific patterns demonstrated

## Development Setup

This project uses Python 3.13+ with `uv` for dependency management.

### Dependencies

- **unsloth**: For efficient LLaMA model fine-tuning
- **ruff**: For code linting and formatting

### Running the Code

```bash
# Generate training data from markdown files
python main.py

# Train the model (requires GPU)
python train_model.py

# Or use uv to run
uv run main.py
uv run train_model.py
```

### Code Quality

```bash
# Lint code with ruff
ruff check .

# Format code with ruff
ruff format .
```

## Training Workflow

The complete workflow to create a code quality advisor model:

1. **Add training data**: Create markdown files in `best-practices/` following the format:
   - `## Principle:` sections with good/bad code examples
   - Each example includes code and explanations

2. **Generate training data**: Run `python main.py` to convert markdown to ChatML format
   - Creates `training_data.jsonl` with user/assistant message pairs
   - Generates 3 variations per principle (code review, best practice, comparison)

3. **Fine-tune model**: Run `python train_model.py` to:
   - Load a base LLaMA model (default: Llama-3.2-3B-Instruct)
   - Apply LoRA fine-tuning on the training data
   - Export to GGUF format for Ollama
   - Generate Modelfile for deployment

4. **Deploy to Ollama**:
   ```bash
   cd model_gguf
   ollama create code-advisor -f Modelfile
   ollama run code-advisor "Review this code..."
   ```

## Key Files

- **main.py**: Converts markdown best practices to training data (ChatML format)
- **train_model.py**: Fine-tunes LLaMA model using Unsloth and exports for Ollama
- **training_data.jsonl**: Generated training data (not committed to git)
- **best-practices/*.md**: Source markdown files with good/bad code examples

## Future Enhancements

- Evaluation tools to test model quality on code review tasks
- Support for `docs/` directory to include documentation in training
- Support for `examples/` directory for standalone code examples
- Validation sets to measure improvement in code quality advice
