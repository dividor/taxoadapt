#!/usr/bin/env python3
"""
Sample humanitarian evaluation data with balanced sampling across agencies.

This script reads an Excel file and creates a balanced sample across all agencies
that have more than 10 reports, ensuring proportional representation.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def sample_humanitarian_data(
    excel_path,
    tab_name,
    title_field,
    abstract_field,
    agency_field,
    sample_size,
    min_reports=10,
    random_seed=42,
    output_path=None
):
    """
    Sample data with balanced representation across agencies.
    
    Args:
        excel_path: Path to Excel file
        tab_name: Sheet/tab name to read
        title_field: Column name for titles
        abstract_field: Column name for abstracts
        agency_field: Column name for agency/organization
        sample_size: Total number of samples to select
        min_reports: Minimum reports per agency to include (default: 10)
        random_seed: Random seed for reproducibility (default: 42)
        output_path: Output Excel path (if None, creates <input>_sampled.xlsx)
    
    Returns:
        DataFrame with sampled data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    print(f"Reading Excel file: {excel_path}")
    print(f"  Tab: {tab_name}")
    print(f"  Title field: {title_field}")
    print(f"  Abstract field: {abstract_field}")
    print(f"  Agency field: {agency_field}")
    print(f"  Random seed: {random_seed}")
    
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=tab_name)
    
    print(f"\nTotal records in dataset: {len(df)}")
    
    # Check required columns exist
    required_cols = [title_field, abstract_field, agency_field]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing data in key fields
    initial_count = len(df)
    df = df.dropna(subset=[title_field, abstract_field, agency_field])
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} rows with missing data")
    
    # Count reports per agency
    agency_counts = df[agency_field].value_counts()
    print(f"\nTotal agencies: {len(agency_counts)}")
    
    # Filter agencies with at least min_reports
    eligible_agencies = agency_counts[agency_counts >= min_reports]
    print(f"Agencies with >= {min_reports} reports: {len(eligible_agencies)}")
    
    if len(eligible_agencies) == 0:
        raise ValueError(f"No agencies have >= {min_reports} reports. Consider lowering --min_reports")
    
    # Filter dataframe to only include eligible agencies
    df_eligible = df[df[agency_field].isin(eligible_agencies.index)]
    print(f"Total eligible records: {len(df_eligible)}")
    
    # Display agency distribution
    print(f"\nEligible agency distribution:")
    for agency, count in eligible_agencies.head(10).items():
        print(f"  {agency}: {count} reports")
    if len(eligible_agencies) > 10:
        print(f"  ... and {len(eligible_agencies) - 10} more agencies")
    
    # Calculate sampling proportions
    total_eligible = len(df_eligible)
    if sample_size > total_eligible:
        print(f"\nWARNING: Requested sample size ({sample_size}) exceeds eligible records ({total_eligible})")
        print(f"Using all {total_eligible} eligible records instead")
        sample_size = total_eligible
    
    # Stratified sampling: sample proportionally from each agency
    sampled_dfs = []
    
    for agency in eligible_agencies.index:
        agency_df = df_eligible[df_eligible[agency_field] == agency]
        agency_proportion = len(agency_df) / total_eligible
        agency_sample_size = int(np.ceil(sample_size * agency_proportion))
        
        # Don't sample more than available
        agency_sample_size = min(agency_sample_size, len(agency_df))
        
        # Random sample from this agency
        agency_sample = agency_df.sample(n=agency_sample_size, random_state=random_seed)
        sampled_dfs.append(agency_sample)
    
    # Combine all samples
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we oversampled (due to rounding up), randomly remove excess
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=random_seed)
    
    # Shuffle the final sample
    sampled_df = sampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"SAMPLING COMPLETE")
    print(f"{'='*60}")
    print(f"Final sample size: {len(sampled_df)}")
    print(f"\nSample distribution by agency:")
    sample_agency_counts = sampled_df[agency_field].value_counts()
    for agency, count in sample_agency_counts.head(15).items():
        original_count = eligible_agencies[agency]
        percentage = (count / original_count) * 100
        print(f"  {agency}: {count}/{original_count} ({percentage:.1f}%)")
    if len(sample_agency_counts) > 15:
        print(f"  ... and {len(sample_agency_counts) - 15} more agencies")
    
    # Save to Excel
    if output_path is None:
        input_path = Path(excel_path)
        output_path = input_path.parent / f"{input_path.stem}_sampled{input_path.suffix}"
    
    sampled_df.to_excel(output_path, sheet_name=tab_name, index=False)
    print(f"\nSampled data saved to: {output_path}")
    
    return sampled_df


def main():
    parser = argparse.ArgumentParser(
        description='Sample humanitarian evaluation data with balanced representation across agencies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sample_humanitarian_data.py \\
    --dataset_sheet "data/evaluations.xlsx" \\
    --dataset_sheet_tabname "PDF Metadata" \\
    --dataset_title_fieldname "Title" \\
    --dataset_abstract_fieldname "Abstractive Summary (map reduced)" \\
    --agency_fieldname "Agency" \\
    --sample_size 100 \\
    --min_reports 10 \\
    --random_seed 42
        """
    )
    
    parser.add_argument('--dataset_sheet', type=str, required=True,
                       help='Path to Excel file')
    parser.add_argument('--dataset_sheet_tabname', type=str, required=True,
                       help='Excel sheet/tab name to read from')
    parser.add_argument('--dataset_title_fieldname', type=str, required=True,
                       help='Column name for paper titles')
    parser.add_argument('--dataset_abstract_fieldname', type=str, required=True,
                       help='Column name for paper abstracts')
    parser.add_argument('--agency_fieldname', type=str, required=True,
                       help='Column name for agency/organization')
    parser.add_argument('--sample_size', type=int, required=True,
                       help='Total number of samples to select')
    parser.add_argument('--min_reports', type=int, default=10,
                       help='Minimum reports per agency to include (default: 10)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output Excel path (default: <input>_sampled.xlsx)')
    
    args = parser.parse_args()
    
    # Run sampling
    sample_humanitarian_data(
        excel_path=args.dataset_sheet,
        tab_name=args.dataset_sheet_tabname,
        title_field=args.dataset_title_fieldname,
        abstract_field=args.dataset_abstract_fieldname,
        agency_field=args.agency_fieldname,
        sample_size=args.sample_size,
        min_reports=args.min_reports,
        random_seed=args.random_seed,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()

