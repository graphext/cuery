# LLM Mentions Auditor

Generates localized commercial-intent prompts, executes them across multiple LLMs with repetitions, detects brand mentions by name and URL, and outputs a coverage matrix suitable for benchmarking.

## Input

Key fields:
- `urls`: Company URLs (seed brands)
- `expand_competitors`: Guess simple competitors to compare (demo heuristic)
- `sector`, `market`, `language`: Context for prompt generation
- `n_prompts`: Number of prompts to generate
- `llms`: Target LLM identifiers (e.g., `openai:gpt-4o`)
- `repetitions`: Times to run each prompt per LLM
- `brand_overrides`: Optional brand alias/domain overrides

## Output

Artifacts in Key-Value Store:
- `brands.json`: normalized brand objects
- `prompts.json`: generated prompts
- `runs.json`: raw LLM responses
- `stats.json`: basic coverage stats
- `notes.json`: process notes

Dataset rows contain per-prompt mention booleans for each `(LLM x brand)` pair with `.by_name` and `.by_url` suffixes.


