#!/usr/bin/env python3
"""
Advanced script to generate jobs for each sector/subsector combination from DIRCE data.
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
from cuery import task
from cuery.response import ResponseModel
from cuery.prompt import Prompt
from pydantic import Field

# Set up paths
GDRIVE = Path("~/Library/CloudStorage/GoogleDrive-victoriano@graphext.com/Shared drives/Solutions").expanduser()
DATA_DIR = GDRIVE / "Research/future_of_work/inputs/ine_dirce_aggregated_by_activity.parquet"
OUTPUT_DIR = GDRIVE / "Research/future_of_work/outputs"

# ============================
# Response Models
# ============================

class Job(ResponseModel):
    """Individual job with automation potential."""
    job_role: str = Field(
        description="Name of the job role (job title, less than 50 characters)",
        min_length=5,
        max_length=50,
    )
    job_description: str = Field(
        description="A short description of the job role (less than 200 characters)",
        min_length=20,
        max_length=200,
    )
    job_automation_potential: int = Field(
        description="A score from 1 to 10 indicating the job's potential for automation",
        ge=0,
        le=10,
    )
    job_automation_reason: str = Field(
        description=(
            "A short explanation of no more than 10 words, of why the job "
            "is likely to be automatable with the given potential score."
        ),
        min_length=30,
        max_length=300,
    )


class Jobs(ResponseModel):
    """List of jobs for a sector/subsector combination."""
    jobs: list[Job] = Field(
        description=(
            "A list of jobs with their AI automation potential and reasons for that potential"
        ),
        min_length=0,
    )

# ============================
# Prompt Definition
# ============================

DIRCE_JOBS_PROMPT = Prompt(
    messages=[
        {
            "role": "system",
            "content": (
                "You're an analyst at the Spanish 'Instituo Nacional de Estadística' (INE) analyzing "
                "data from its 'Directorio Central de Empresas' (DIRCE). Your objective is to analyze "
                "groups of companies, identified by a sector ('Sector') and a corresponding main activity "
                "('Subsector') in order to identify jobs within those companies that are likely to "
                "be automatable by AI. Both 'Sector' and 'Subsector' are provided in Spanish and may "
                "include numeric IDs that you can ignore if you don't understand them. Always respond in English. "
                "Only consider jobs that are computer- or paper-based and can be automated by AI using software "
                "(don't include jobs automatable by robots or other physical means)."
            )
        },
        {
            "role": "user",
            "content": (
                "Please analyze the following jobs sector and identify jobs that are automatable by AI software.\n\n"
                "Sector: {{sector}}\n\n"
                "Subsector: {{subsector}}"
            )
        }
    ],
    required=["sector", "subsector"]
)

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

# Configuration variables for easy testing
MODEL_NAME = "gpt-4o-mini"  # Options: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
TEST_SAMPLE_SIZE = 2        # Number of sectors to process in test mode (can be changed to 10, 20, etc.)

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
        
    def load_progress(self) -> dict:
        """Load progress from file if exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"processed_indices": [], "results_file": None}
    
    def save_progress(self, results_file: str):
        """Save current progress to file."""
        progress = {
            "processed_indices": list(self.processed_indices),
            "results_file": results_file,
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
                model=MODEL_NAME,
                n_concurrent=self.n_concurrent
            )
            
            # Convert to DataFrame and add batch indices
            jobs_df = jobs_result.to_pandas(explode=True)
            
            # Mark these indices as processed
            self.processed_indices.update(batch_indices)
            
            return jobs_df
            
        except Exception as e:
            print(f"\nError processing batch: {e}")
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
                results_file = progress["results_file"]
            else:
                results_file = str(OUTPUT_DIR / f"dirce_generated_jobs_{timestamp}.csv")
        else:
            results_file = str(OUTPUT_DIR / f"dirce_generated_jobs_{timestamp}.csv")
        
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
            print(f"Total jobs generated: {len(final_df)}")
            print(f"Total sectors processed: {len(self.processed_indices)}")
            print(f"Average jobs per sector: {len(final_df) / len(self.processed_indices):.1f}")
            print(f"\nJob automation potential distribution:")
            print(final_df['job_automation_potential'].value_counts().sort_index())
            
            print(f"\nResults saved to:")
            
            # Clean up progress file on successful completion
            if len(self.processed_indices) == len(context_df):
                self.progress_file.unlink(missing_ok=True)
                print("\nAll sectors processed successfully!")

async def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate jobs from DIRCE data")
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