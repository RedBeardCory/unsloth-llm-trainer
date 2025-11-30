import json
import re
from pathlib import Path
from typing import Dict, List


def parse_best_practices_md(md_content: str) -> List[Dict]:
    """
    Parse a best practices markdown file and convert it to unsloth training format.

    Expected markdown structure:
    - ## Principle: <description>
    - ### Bad Practice: <title>
    - ```python code block ```
    - **Why this is bad:** explanation
    - ### Good Practice: <title>
    - ```python code block ```
    - **Why this is good:** explanation

    Returns a list of training examples in ChatML format.
    """
    training_examples = []

    # Split by "## Principle:" sections
    principle_sections = re.split(r"## Principle:", md_content)

    for section in principle_sections[1:]:  # Skip first empty section
        lines = section.strip().split("\n")
        if not lines:
            continue

        principle = lines[0].strip()

        # Extract bad practice section
        bad_match = re.search(
            r"### Bad Practice:.*?\n(.*?)```python\n(.*?)```\n.*?\*\*Why this is bad:\*\*\n(.*?)(?=###|\Z)",
            section,
            re.DOTALL,
        )

        # Extract good practice section
        good_match = re.search(
            r"### Good Practice:.*?\n(.*?)```python\n(.*?)```\n.*?\*\*Why this is good:\*\*\n(.*?)(?=##|\Z)",
            section,
            re.DOTALL,
        )

        if bad_match and good_match:
            bad_code = bad_match.group(2).strip()
            bad_reasons = bad_match.group(3).strip()
            good_code = good_match.group(2).strip()
            good_reasons = good_match.group(3).strip()

            # Create training example: Ask to identify problems
            training_examples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Review this code and identify any issues:\n\n```python\n{bad_code}\n```",
                        },
                        {
                            "role": "assistant",
                            "content": f"This code has issues related to: {principle}\n\n{bad_reasons}\n\nHere's an improved version:\n\n```python\n{good_code}\n```\n\n{good_reasons}",
                        },
                    ]
                }
            )

            # Create training example: Ask for best practice
            training_examples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"What is the best practice for: {principle}",
                        },
                        {
                            "role": "assistant",
                            "content": f"{good_reasons}\n\nHere's an example:\n\n```python\n{good_code}\n```",
                        },
                    ]
                }
            )

            # Create training example: Direct comparison
            training_examples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Show me good vs bad examples for: {principle}",
                        },
                        {
                            "role": "assistant",
                            "content": f"**Bad Practice:**\n```python\n{bad_code}\n```\n\n{bad_reasons}\n\n**Good Practice:**\n```python\n{good_code}\n```\n\n{good_reasons}",
                        },
                    ]
                }
            )

    return training_examples


def convert_markdown_to_training_data(input_path: str, output_path: str):
    """
    Convert a markdown best practices file to unsloth training format (ChatML).

    Args:
        input_path: Path to the input markdown file
        output_path: Path to save the output JSONL file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    training_examples = parse_best_practices_md(md_content)

    # Save as JSONL (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Converted {len(training_examples)} training examples to {output_path}")
    return training_examples


def process_all_best_practices(
    input_dir: str = "best-practices", output_file: str = "training_data.jsonl"
):
    """
    Process all markdown files in the best-practices directory and combine them
    into a single training dataset.

    Args:
        input_dir: Directory containing markdown files
        output_file: Output JSONL file path
    """
    all_examples = []
    best_practices_path = Path(input_dir)

    if not best_practices_path.exists():
        print(f"Error: {input_dir} directory not found")
        return

    md_files = list(best_practices_path.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    for md_file in md_files:
        print(f"Processing {md_file.name}...")
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        examples = parse_best_practices_md(md_content)
        all_examples.extend(examples)
        print(f"  Generated {len(examples)} examples")

    # Save all examples to a single file
    with open(output_file, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\nTotal: {len(all_examples)} training examples saved to {output_file}")
    return all_examples


def main():
    print("Hello from llama-experiments!")
    print("\nProcessing best practices markdown files...")
    process_all_best_practices()


if __name__ == "__main__":
    main()
