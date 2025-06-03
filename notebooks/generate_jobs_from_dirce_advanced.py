#!/usr/bin/env python3
"""
Advanced script to generate jobs for each sector/subsector combination from DIRCE data.
Features: batch processing, progress tracking, resume capability, and error handling.
"""

import asyncio
from pathlib import Path
import polars as pl
import pandas as pd
from datetime import datetime
import json
from typing import Optional
import argparse

from cuery import task
from cuery.work import DirceJobs
from tqdm import tqdm

# Set up paths
GDRIVE = Path("~/Library/CloudStorage/GoogleDrive-victoriano@graphext.com/Shared drives/Solutions").expanduser()
DATA_DIR = GDRIVE / "Research/future_of_work/inputs/ine_dirce_aggregated_by_activity.parquet"
OUTPUT_DIR = GDRIVE / "Research/future_of_work/outputs"

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
                model="gpt-4o-mini",
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
        
        # Convert to pandas
        context_df = df_filtered.select(["sector", "subsector"]).to_pandas()
        context_df.reset_index(drop=True, inplace=True)
        
        # Limit data in test mode
        if self.test_mode:
            context_df = context_df.head(5)
            print("TEST MODE: Processing only first 5 sectors")
        
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
                        help="Test mode: process only first 5 sectors")
    parser.add_argument("--format", type=str, default='csv', choices=['csv', 'parquet', 'both'],
                        help="Format for saving results (default: csv)")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Adjust for test mode
    if args.test:
        args.batch_size = 5
        print("Running in TEST mode - processing only first 5 sectors")
    
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