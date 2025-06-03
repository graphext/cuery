#!/usr/bin/env python3
"""
Advanced script to generate specific automatable tasks across occupations from DIRCE data.
For each sector/subsector, generates multiple tasks with occupation info, automation scores, and current tools.
Features: batch processing, progress tracking, resume capability, and error handling.
Now self-contained with all models and prompts included.
"""

import asyncio
from pathlib import Path
import polars as pl
import pandas as pd
from datetime import datetime
import json
from typing import Optional
import argparse
from tqdm import tqdm

# Cuery imports
import instructor
from cuery import task
from cuery.response import ResponseModel
from cuery.prompt import Prompt
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import Field

# Set up paths
GDRIVE = Path("~/Library/CloudStorage/GoogleDrive-victoriano@graphext.com/Shared drives/Solutions").expanduser()
DATA_DIR = GDRIVE / "Research/future_of_work/inputs/ine_dirce_aggregated_by_activity.parquet"
OUTPUT_DIR = GDRIVE / "Research/future_of_work/outputs"

# ============================
# Response Models
# ============================

class Task(ResponseModel):
    """Individual task within an occupation."""
    task_name: str = Field(
        description="Name of the specific task or job to be done (less than 80 characters)",
        min_length=10,
        max_length=80,
    )

    task_automation_potential: int = Field(
        description="A score from 1 to 10 indicating this specific task's potential for automation",
        ge=0,
        le=10,
    )

    current_products: list[str] = Field(
        description="List of current software products/tools used to perform this task",
        min_length=1,
        max_length=8,
    )

    task_description: str = Field(
        description="Detailed description of what this task involves (less than 300 characters)",
        min_length=30,
        max_length=300,
    )

    task_automation_reason: str = Field(
        description="A short explanation of why this task is automatable with the given score",
        min_length=20,
        max_length=200,
    )

class Occupation(ResponseModel):
    """An occupation/job role with its associated tasks."""
    occupation_name: str = Field(
        description="Name of the occupation/job role (less than 60 characters)",
        min_length=5,
        max_length=60,
    )

    tasks: list[Task] = Field(
        description="List of specific tasks performed in this occupation",
        min_length=3,
        max_length=10,
    )

class Jobs(ResponseModel):
    """List of occupations for a sector/subsector."""
    occupations: list[Occupation] = Field(
        description="List of all occupations/job roles in this sector/subsector",
        min_length=1,
        max_length=30,
    )

# ============================
# Prompt Definition
# ============================

DIRCE_JOBS_PROMPT = Prompt(
    messages=[
        {
            "role": "system",
            "content": (
                "You're an analyst at the Spanish 'Instituto Nacional de Estadística' (INE) analyzing "
                "data from its 'Directorio Central de Empresas' (DIRCE). Your objective is to analyze "
                "groups of companies, identified by a sector ('Sector') and a corresponding main activity "
                "('Subsector') in order to identify relevant occupations and their specific automatable tasks. "
                "Both 'Sector' and 'Subsector' are provided in Spanish and may "
                "include numeric IDs that you can ignore if you don't understand them. Always respond in English. "
                "Only consider tasks that are computer- or paper-based and can be automated by AI using software "
                "(don't include tasks automatable by robots or other physical means). For each task, also identify "
                "the current software products or tools commonly used to perform that task."
            )
        },
        {
            "role": "user",
            "content": (
                "Please analyze the following sector:\n\n"
                "1. First, identify relevant occupations/job roles that would typically exist in this sector/subsector\n"
                "2. For EACH occupation, provide specific automatable tasks they perform\n\n"
                "Structure your response hierarchically:\n"
                "- Occupation 1: [name]\n"
                "  - Task: [task details, automation score, current tools]\n"
                "  - Task: [task details, automation score, current tools]\n"
                "  - Task: [task details, automation score, current tools]\n"
                "  - ... (as many tasks as relevant for this occupation)\n"
                "- Occupation 2: [name]\n"
                "  - Task: [task details, automation score, current tools]\n"
                "  - Task: [task details, automation score, current tools]\n"
                "  - ... (as many tasks as relevant for this occupation)\n"
                "- ... (continue for all relevant occupations)\n\n"
                "Generate as many occupations and tasks as you consider relevant for the sector.\n"
                "For each task, include automation potential (1-10) and current products/tools used.\n\n"
                "Sector: {{sector}}\n\n"
                "Subsector: {{subsector}}"
            )
        }
    ],
    required=["sector", "subsector"]
)

# ============================
# Configuration
# ============================

# Configuration variables for easy testing
MODEL_NAME = "claude-3-haiku-20240307"  # Options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
TEST_SAMPLE_SIZE = 1        # Number of sectors to process in test mode (can be changed to 10, 20, etc.)

def is_openai_model(model_name: str) -> bool:
    """Check if the model is from OpenAI (cost tracking supported)."""
    openai_prefixes = ["gpt-", "text-", "davinci", "curie", "babbage", "ada"]
    return any(model_name.startswith(prefix) for prefix in openai_prefixes)

def get_client_for_model(model_name: str):
    """Get the appropriate client for the model."""
    if is_openai_model(model_name):
        from openai import AsyncOpenAI
        return instructor.from_openai(AsyncOpenAI())
    elif model_name.startswith("claude-"):
        return instructor.from_anthropic(AsyncAnthropic())
    else:
        # Default to OpenAI
        from openai import AsyncOpenAI
        return instructor.from_openai(AsyncOpenAI())

# Get client based on model
CLIENT = get_client_for_model(MODEL_NAME)

print(f"Using model: {MODEL_NAME}")
print(f"Cost tracking: {'Available' if is_openai_model(MODEL_NAME) else 'Not available (OpenAI models only)'}")

# ============================
# Task Definition
# ============================

# Create the DirceJobs task locally
DirceJobs = task.Task(
    name="DirceJobs", 
    prompt=DIRCE_JOBS_PROMPT, 
    response=Jobs
)

# ============================
# Job Generator Class
# ============================


class JobGenerator:
    """Class to handle job generation with progress tracking and resume capability."""
    
    def __init__(self, batch_size: int = 25, n_concurrent: int = 10, test_mode: bool = False, format: str = 'csv'):
        self.batch_size = batch_size
        self.n_concurrent = n_concurrent
        self.test_mode = test_mode
        self.format = format
        self.progress_file = OUTPUT_DIR / "dirce_jobs_progress.json"
        self.results = []
        self.processed_indices = set()
        self.total_cost = 0.0  # Track total cost across all batches
        
    def load_progress(self) -> dict:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                # Load previous cost if available
                if 'total_cost' in progress_data:
                    self.total_cost = progress_data['total_cost']
                return progress_data
        return {"processed_indices": [], "results_file": None, "total_cost": 0.0}
    
    def save_progress(self, results_file: str):
        """Save current progress to file."""
        progress = {
            "processed_indices": list(self.processed_indices),
            "results_file": results_file,
            "total_cost": self.total_cost,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    async def process_batch(self, batch_df: pd.DataFrame, batch_indices: list) -> pd.DataFrame:
        """Process a single batch of sector/subsector combinations."""
        try:
            # Run DirceJobs task
            jobs_result = await DirceJobs(
                context=batch_df,
                client=CLIENT,
                model=MODEL_NAME,
                n_concurrent=self.n_concurrent
            )
            
            # Track cost if available (only for OpenAI models)
            batch_cost = 0.0
            if is_openai_model(MODEL_NAME):
                try:
                    # Try different possible cost attributes
                    if hasattr(jobs_result, 'cost'):
                        batch_cost = jobs_result.cost
                    elif hasattr(jobs_result, 'total_cost'):
                        batch_cost = jobs_result.total_cost
                    elif hasattr(jobs_result, 'usage'):
                        # Call usage as a method - returns a DataFrame
                        usage_df = jobs_result.usage()
                        
                        if not usage_df.empty and 'cost' in usage_df.columns:
                            batch_cost = usage_df['cost'].sum()
                            total_prompt_tokens = usage_df['prompt'].sum() if 'prompt' in usage_df.columns else 0
                            total_completion_tokens = usage_df['completion'].sum() if 'completion' in usage_df.columns else 0
                            print(f"Tokens used: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion = {total_prompt_tokens + total_completion_tokens:,} total")
                    
                    if batch_cost > 0:
                        self.total_cost += batch_cost
                        print(f"Batch cost: ${batch_cost:.4f} | Total cost so far: ${self.total_cost:.4f}")
                        
                except Exception as e:
                    print(f"Note: Could not retrieve cost information: {e}")
            else:
                print(f"Cost tracking not available for {MODEL_NAME} (only supported for OpenAI models)")
            
            # Convert to DataFrame and handle nested structure
            # The structure is already partially flattened: each row is an occupation with tasks
            jobs_df = jobs_result.to_pandas()
            
            # Now we need to flatten the tasks within each occupation
            all_tasks = []
            
            for _, row in jobs_df.iterrows():
                sector = row['sector']
                subsector = row['subsector']
                occupation_name = row['occupation_name']
                
                # Iterate through tasks in this occupation
                for task in row['tasks']:
                    task_row = {
                        'sector': sector,
                        'subsector': subsector,
                        'occupation': occupation_name,
                        'task_name': task.task_name,
                        'task_description': task.task_description,
                        'task_automation_potential': task.task_automation_potential,
                        'task_automation_reason': task.task_automation_reason,
                        'current_products': task.current_products
                    }
                    all_tasks.append(task_row)
            
            # Create flattened DataFrame
            flattened_df = pd.DataFrame(all_tasks)
            
            # Mark these indices as processed
            self.processed_indices.update(batch_indices)
            
            return flattened_df
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    async def generate_jobs(self, resume: bool = False):
        """Main method to generate jobs with batch processing."""
        
        # Read the data
        print(f"Reading data from: {DATA_DIR}")
        df = pl.read_parquet(DATA_DIR)
        
        # Rename columns
        df = df.rename({
            "Division": "sector",
            "Actividad principal": "subsector"
        })
        
        # Filter out nulls
        df_filtered = df.filter(
            (pl.col("sector").is_not_null()) & 
            (pl.col("subsector").is_not_null())
        )
        
        # Convert to pandas - include employee count for sorting
        context_df = df_filtered.select(["sector", "subsector", "Estimated_Employees_2024"]).to_pandas()
        
        # Sort by employee count in descending order (largest sectors first)
        context_df = context_df.sort_values("Estimated_Employees_2024", ascending=False)
        context_df.reset_index(drop=True, inplace=True)
        
        # Limit data in test mode (now gets the largest sectors)
        if self.test_mode:
            context_df_top = context_df.head(TEST_SAMPLE_SIZE)
            print(f"TEST MODE: Processing top {TEST_SAMPLE_SIZE} sectors by employee count")
            print(f"Top {TEST_SAMPLE_SIZE} sectors:")
            for i, row in context_df_top.iterrows():
                print(f"  {i+1}. {row['sector']} / {row['subsector']} ({row['Estimated_Employees_2024']:,.0f} employees)")
            context_df = context_df_top
        
        # Remove employee count column (keep only sector and subsector for processing)
        context_df = context_df[["sector", "subsector"]].copy()
        
        print(f"Total sectors to process: {len(context_df)}")
        
        # Handle resume
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if resume:
            progress = self.load_progress()
            self.processed_indices = set(progress["processed_indices"])
            if progress["results_file"] and Path(progress["results_file"]).exists():
                # Load existing results
                existing_df = pd.read_csv(progress["results_file"])
                self.results.append(existing_df)
                print(f"Resuming from previous run. Already processed: {len(self.processed_indices)} sectors")
                if self.total_cost > 0:
                    print(f"Previous total cost: ${self.total_cost:.4f}")
                elif not is_openai_model(MODEL_NAME):
                    print("Cost tracking not available for this model")
                results_file = progress["results_file"]
                results_file = str(OUTPUT_DIR / f"dirce_generated_tasks_{timestamp}.csv")
        else:
            results_file = str(OUTPUT_DIR / f"dirce_generated_tasks_{timestamp}.csv")
        
        # Process in batches
        total_batches = (len(context_df) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(context_df), desc="Processing sectors") as pbar:
            # Update progress bar for already processed items
            pbar.update(len(self.processed_indices))
            
            for i in range(0, len(context_df), self.batch_size):
                batch_indices = list(range(i, min(i + self.batch_size, len(context_df))))
                
                # Skip if already processed
                if all(idx in self.processed_indices for idx in batch_indices):
                    continue
                
                # Filter to only unprocessed indices in this batch
                unprocessed_indices = [idx for idx in batch_indices if idx not in self.processed_indices]
                if not unprocessed_indices:
                    continue
                
                batch_df = context_df.iloc[unprocessed_indices]
                
                print(f"\nProcessing batch: {len(unprocessed_indices)} sectors")
                
                # Process batch
                batch_results = await self.process_batch(batch_df, unprocessed_indices)
                
                if not batch_results.empty:
                    self.results.append(batch_results)
                    
                    # Save intermediate results
                    all_results = pd.concat(self.results, ignore_index=True)
                    all_results.to_csv(results_file, index=False)
                    
                    # Save progress
                    self.save_progress(results_file)
                    
                    pbar.update(len(unprocessed_indices))
                    
                # Small delay between batches to avoid rate limits
                await asyncio.sleep(1)
        
        # Final save
        if self.results:
            final_df = pd.concat(self.results, ignore_index=True)
            
            # Save based on format preference
            if self.format in ['csv', 'both']:
                final_df.to_csv(results_file, index=False)
                print(f"  CSV: {results_file}")
            
            if self.format in ['parquet', 'both']:
                parquet_file = results_file.replace('.csv', '.parquet')
                final_df.to_parquet(parquet_file, index=False)
                print(f"  Parquet: {parquet_file}")
            
            # Print summary
            print("\n=== Summary Statistics ===")
            print(f"Total tasks generated: {len(final_df)}")
            print(f"Total sectors processed: {len(self.processed_indices)}")
            
            # Count unique occupations
            if 'occupation' in final_df.columns:
                unique_occupations = final_df['occupation'].nunique()
                print(f"Total unique occupations: {unique_occupations}")
                print(f"Average occupations per sector: {unique_occupations / len(self.processed_indices):.1f}")
                print(f"Average tasks per occupation: {len(final_df) / unique_occupations:.1f}")
                print(f"Average tasks per sector: {len(final_df) / len(self.processed_indices):.1f}")
            
            # Cost information
            if self.total_cost > 0:
                print(f"\n=== Cost Information ===")
                print(f"Total cost: ${self.total_cost:.4f}")
                print(f"Average cost per sector: ${self.total_cost / len(self.processed_indices):.4f}")
                print(f"Average cost per task: ${self.total_cost / len(final_df):.4f}")
            elif not is_openai_model(MODEL_NAME):
                print(f"\n=== Cost Information ===")
                print(f"Cost tracking not available for {MODEL_NAME}")
                print("Cost tracking is currently only supported for OpenAI models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.)")
            
            print(f"\nTask automation potential distribution:")
            print(final_df['task_automation_potential'].value_counts().sort_index())
            
            print(f"\nResults saved to:")
            
            # Clean up progress file on successful completion
            if len(self.processed_indices) == len(context_df):
                self.progress_file.unlink(missing_ok=True)
                print("\nAll sectors processed successfully!")

async def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate automatable tasks from DIRCE data")
    parser.add_argument("--batch-size", type=int, default=25, 
                        help="Number of sectors to process in each batch (default: 25)")
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Number of concurrent API requests (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run if interrupted")
    parser.add_argument("--test", action="store_true",
                        help=f"Test mode: process top {TEST_SAMPLE_SIZE} sectors by employee count")
    parser.add_argument("--format", type=str, default='csv', choices=['csv', 'parquet', 'both'],
                        help="Format for saving results (default: csv)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Adjust for test mode
    if args.test:
        args.batch_size = TEST_SAMPLE_SIZE
        print(f"Running in TEST mode - processing top {TEST_SAMPLE_SIZE} sectors by employee count")
    
    # Create generator and run
    generator = JobGenerator(
        batch_size=args.batch_size,
        n_concurrent=args.concurrent,
        test_mode=args.test,
        format=args.format
    )
    
    await generator.generate_jobs(resume=args.resume)

if __name__ == "__main__":
    asyncio.run(main()) 