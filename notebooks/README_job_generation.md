# DIRCE Task Generation Script

This script generates specific automatable tasks across occupations from DIRCE (Directorio Central de Empresas) data using AI.

## Overview

The script reads the DIRCE aggregated activity data and uses AI (via the cuery library) to generate relevant job roles for each combination of Division (sector) and Actividad principal (subsector).

## Script: `generate_jobs_from_dirce_advanced.py`

**Self-contained script** with all models, prompts, and task definitions included for easy customization.

Features:
- Batch processing to handle large datasets
- Progress tracking and resume capability
- Better error handling
- Command-line arguments
- Test mode for quick sampling
- **All models and prompts included in the file for easy editing**

## Key Components (Now Included in Script)

### Response Models
- **Job**: Individual job with automation potential (fields: job_role, job_description, job_automation_potential, job_automation_reason)
- **Jobs**: Container for multiple jobs per sector/subsector

### Prompt
The prompt instructs the AI to analyze Spanish sectors/subsectors and identify computer/paper-based jobs that can be automated by AI software. You can easily modify the prompt directly in the script to:
- Change the analysis criteria
- Adjust the language or tone
- Add or remove constraints
- Modify the scoring system

### Task
The DirceJobs task combines the prompt and response models, configured to use the Jobs response model.

## Customization Guide

Since all components are now in the script, you can easily customize:

### Configuration Variables (Quick Setup)
At the top of the script, you'll find easy-to-modify variables:

```python
MODEL_NAME = "gpt-4o-mini"  # Change model here
TEST_SAMPLE_SIZE = 5        # Change test sample size here
```

**Model Options:**
- `"gpt-4o"` - Highest quality, slower, more expensive
- `"gpt-4o-mini"` - Good balance (default)
- `"gpt-3.5-turbo"` - Fastest, cheapest
- `"claude-3-sonnet-20240229"` - Anthropic's balanced model
- `"claude-3-haiku-20240307"` - Anthropic's fast model

**Sample Size Examples:**
- `TEST_SAMPLE_SIZE = 3` - Quick test with top 3 sectors
- `TEST_SAMPLE_SIZE = 10` - More comprehensive test
- `TEST_SAMPLE_SIZE = 20` - Broader analysis

### 1. Modify the Prompt
Find the `DIRCE_JOBS_PROMPT` variable and adjust:
- The system message to change the AI's role or perspective
- The user message template to request different information
- Add more context or examples

### 2. Adjust Response Models
Modify the `Job` class to:
- Change field names or descriptions
- Add new fields (e.g., required_skills, tools_used)
- Adjust validation rules (min/max lengths, score ranges)

### 3. Change Scoring
The `job_automation_potential` field currently uses 0-10. You can:
- Change the range (e.g., 1-5 or 0-100)
- Add categories instead of numbers
- Include multiple scoring dimensions

### 4. Advanced Customization
For more complex changes, you can modify the response models, add new fields, or change validation rules directly in the code.

## Quick Start - Test with Top 5 Sectors by Employment

To run a quick test with the 5 largest sectors by employee count from your input file:

```bash
cd notebooks
uv run python generate_jobs_from_dirce_advanced.py --test
```

This will:
- Sort all sectors by `Estimated_Employees_2024` in descending order
- Process only the top 5 sectors with the most employees
- Generate jobs for those 5 largest sector/subsector combinations
- Save results to the output directory
- Complete in under a minute

## Full Usage Options

```bash
cd notebooks

# Test mode (process top 5 sectors by employment) - RECOMMENDED FIRST RUN
uv run python generate_jobs_from_dirce_advanced.py --test

# Test mode with different output formats
uv run python generate_jobs_from_dirce_advanced.py --test --format csv      # default
uv run python generate_jobs_from_dirce_advanced.py --test --format parquet  # more efficient
uv run python generate_jobs_from_dirce_advanced.py --test --format both     # both formats

# Basic run (all sectors)
uv run python generate_jobs_from_dirce_advanced.py

# Custom batch size and concurrency
uv run python generate_jobs_from_dirce_advanced.py --batch-size 50 --concurrent 20

# Resume from previous run if interrupted
uv run python generate_jobs_from_dirce_advanced.py --resume
```

## Command-line Arguments

- `--test`: **Test mode - process top 5 sectors by employment (RECOMMENDED FOR FIRST RUN)**
- `--format`: Output format - `csv` (default), `parquet`, or `both`
- `--batch-size`: Number of sectors to process in each batch (default: 25)
- `--concurrent`: Number of concurrent API requests (default: 10)
- `--resume`: Resume from previous run if interrupted

## Prerequisites

Make sure you have the cuery environment set up with all dependencies:

```bash
# If you haven't already, install dependencies in your project
uv sync
```

## Input Data

The script expects the DIRCE data file at:
```
~/Library/CloudStorage/GoogleDrive-victoriano@graphext.com/Shared drives/Solutions/Research/future_of_work/inputs/ine_dirce_aggregated_by_activity.parquet
```

## Output

Results are saved to:
```
~/Library/CloudStorage/GoogleDrive-victoriano@graphext.com/Shared drives/Solutions/Research/future_of_work/outputs/
```

Output files (based on `--format` option):
- `--format csv` (default): `dirce_generated_jobs_YYYYMMDD_HHMMSS.csv`
- `--format parquet`: `dirce_generated_jobs_YYYYMMDD_HHMMSS.parquet`
- `--format both`: Both CSV and Parquet files

## Generated Data Structure

For each sector/subsector combination, the AI generates multiple specific tasks across different occupations with:
- `sector`: Original sector name from DIRCE
- `subsector`: Original subsector name from DIRCE  
- `occupation`: Name of the occupation this task belongs to
- `occupation_description`: General description of the occupation
- `task_name`: Name of the specific task/job to be done
- `task_description`: Detailed description of what this task involves
- `task_automation_potential`: Score from 0-10 indicating this specific task's automation potential
- `task_automation_reason`: Explanation for the automation potential score
- `current_products`: Array of current software products/tools used to perform this task

## Output Structure

The script now generates **multiple tasks for each occupation**, providing granular insight into automatable work. Each row represents a specific task that can be automated, with details about:

- **Which occupation** it belongs to
- **What tools** are currently used
- **How automatable** it is (0-10 scale)
- **Why** it can be automated

This gives you actionable data about specific work that can be automated rather than just general job categories.

## Resume Capability

The script saves progress in `dirce_jobs_progress.json`. If interrupted:
1. Run with `--resume` flag to continue from where it left off
2. The script will load previous results and skip already processed sectors
3. Progress file is automatically deleted upon successful completion

## Performance Tips

1. **Always start with `--test` mode** to verify everything works
2. Adjust `--batch-size` based on your API limits (lower if you get rate limit errors)
3. Use `--concurrent` to control API request rate (lower if you get rate limit errors)
4. The parquet output format is recommended for better performance when reading results
5. For the full dataset (252 sectors), expect ~6-8 minutes of processing time

## Error Handling

- The script continues processing even if individual batches fail
- Failed batches are logged but don't stop the entire process
- Results are saved incrementally after each batch 