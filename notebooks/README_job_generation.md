# DIRCE Job Generation Script

This script generates job roles from DIRCE (Directorio Central de Empresas) data using AI.

## Overview

The script reads the DIRCE aggregated activity data and uses AI (via the cuery library) to generate relevant job roles for each combination of Division (sector) and Actividad principal (subsector).

## Script: `generate_jobs_from_dirce_advanced.py`

Advanced script with features:
- Batch processing to handle large datasets
- Progress tracking and resume capability
- Better error handling
- Command-line arguments
- Test mode for quick sampling

## Quick Start - Test with Top 5 Rows

To run a quick test with just the first 5 sectors from your input file:

```bash
cd notebooks
uv run python generate_jobs_from_dirce_advanced.py --test
```

This will:
- Process only the first 5 rows of the DIRCE data
- Generate jobs for those 5 sector/subsector combinations
- Save results to the output directory
- Complete in under a minute

## Full Usage Options

```bash
cd notebooks

# Test mode (process only first 5 sectors) - RECOMMENDED FIRST RUN
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

- `--test`: **Test mode - process only first 5 sectors (RECOMMENDED FOR FIRST RUN)**
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

For each sector/subsector combination, the AI generates multiple job roles with:
- `sector`: Original sector name from DIRCE
- `subsector`: Original subsector name from DIRCE  
- `job_role`: Name of the job (e.g., "Data Entry Clerk")
- `job_description`: Brief description of the role
- `job_automation_potential`: Score from 0-10 indicating automation potential
- `job_automation_reason`: Explanation for the automation potential score

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