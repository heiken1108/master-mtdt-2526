import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def get_file_size_info(filename):
    """Get file size and estimate number of rows"""
    size_bytes = os.path.getsize(filename)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    if size_gb > 1:
        print(f"File size: {size_gb:.2f} GB")
    else:
        print(f"File size: {size_mb:.2f} MB")
    
    return size_bytes

def count_total_rows(filename):
    """Quickly count total rows in file"""
    print("Counting total rows...")
    with open(filename, 'r') as f:
        total_rows = sum(1 for line in f) - 1  # subtract header
    print(f"Total rows: {total_rows:,}")
    return total_rows

def sample_csv_large(input_file, output_file, discriminent_col, sample_pct, chunksize=50000):
    """
    Sample large CSV files efficiently using chunks
    """
    print(f"Processing large file: {input_file}")
    get_file_size_info(input_file)
    
    # Count total rows for progress tracking
    total_rows = count_total_rows(input_file)
    estimated_chunks = (total_rows + chunksize - 1) // chunksize
    
    print(f"Processing in chunks of {chunksize:,} rows ({estimated_chunks} chunks estimated)")
    
    # First pass: collect all unique values
    print("\nStep 1: Finding all unique values...")
    unique_pcs = set()
    
    chunk_iterator = pd.read_csv(input_file, chunksize=chunksize)
    
    with tqdm(total=estimated_chunks, desc="Scanning for unique values", unit="chunk") as pbar:
        for chunk in chunk_iterator:
            unique_pcs.update(chunk[discriminent_col].unique())
            pbar.update(1)
    
    unique_pcs = list(unique_pcs)
    number_to_sample = max(1, int(len(unique_pcs) * sample_pct))
    
    print(f"\nFound {len(unique_pcs):,} unique values in '{discriminent_col}'")
    print(f"Sampling {number_to_sample:,} unique values ({sample_pct*100}%)")
    
    # Sample the unique values
    np.random.seed(42)
    sampled_pcs = np.random.choice(unique_pcs, size=number_to_sample, replace=False)
    sampled_set = set(sampled_pcs)  # Convert to set for O(1) lookup
    
    # Second pass: filter and save
    print(f"\nStep 2: Filtering and saving data...")
    
    chunk_iterator = pd.read_csv(input_file, chunksize=chunksize)
    total_filtered_rows = 0
    first_chunk = True
    
    with tqdm(total=estimated_chunks, desc="Filtering chunks", unit="chunk") as pbar:
        for chunk in chunk_iterator:
            # Filter this chunk
            filtered_chunk = chunk[chunk[discriminent_col].isin(sampled_set)]
            
            if len(filtered_chunk) > 0:
                # Save chunk (append after first chunk)
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                
                filtered_chunk.to_csv(output_file, mode=mode, index=False, header=header)
                total_filtered_rows += len(filtered_chunk)
                first_chunk = False
            
            pbar.update(1)
            pbar.set_postfix({
                'filtered_rows': f"{total_filtered_rows:,}",
                'unique_sampled': f"{len(sampled_pcs):,}"
            })
    
    print(f"\nCompleted!")
    print(f"Original rows: {total_rows:,}")
    print(f"Filtered rows: {total_filtered_rows:,} ({total_filtered_rows/total_rows*100:.2f}%)")
    print(f"Unique {discriminent_col} sampled: {len(sampled_pcs):,}")
    print(f"Saved to: {output_file}")
    
    return total_filtered_rows

def sample_csv_regular(input_file, output_file, discriminent_col, sample_pct):
    """Regular version for smaller files"""
    print(f"Reading {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Processing {len(df):,} rows...")
    unique_pcs = df[discriminent_col].unique()
    
    number_to_sample = max(1, int(len(unique_pcs) * sample_pct))
    print(f"Sampling {number_to_sample:,} unique values from {discriminent_col} (out of {len(unique_pcs):,} total)")
    
    np.random.seed(42)
    sampled_pcs = np.random.choice(unique_pcs, size=number_to_sample, replace=False)
    
    print("Filtering rows...")
    tqdm.pandas(desc="Filtering")
    sampled_set = set(sampled_pcs)
    filtered_df = df[df[discriminent_col].progress_apply(lambda x: x in sampled_set)]
    
    print(f"Saving {len(filtered_df):,} rows to {output_file}")
    filtered_df.to_csv(output_file, index=False)
    
    return filtered_df

if __name__ == "__main__":
    percent = 0.01
    column = "customer_ID"
    input_file = "amex_train_data.csv"
    output_file = f"Sample_{percent}_{input_file}"
    
    # Check file size and choose method
    try:
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        
        if file_size_mb > 500:  # If file is larger than 500MB, use chunked method
            print("Large file detected - using chunked processing")
            chunksize = 50000  # Adjust based on your RAM
            result = sample_csv_large(input_file, output_file, column, percent, chunksize)
        else:
            print("Using regular processing")
            result = sample_csv_regular(input_file, output_file, column, percent)
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  {f}")