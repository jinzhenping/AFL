"""
Evaluate prior recommendation accuracy from a CSV file.
This script reads a prior CSV file (id,generate,real) and calculates accuracy metrics.
"""
import os
import argparse
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate prior recommendation accuracy from CSV file')
    parser.add_argument('--prior_file', type=str, required=True, help='Path to prior CSV file (e.g., ./data/prior_mind/SASRec.csv)')
    return parser.parse_args()


def evaluate_prior_accuracy(prior_file):
    """Evaluate accuracy metrics from prior CSV file"""
    
    if not os.path.exists(prior_file):
        raise FileNotFoundError(f"Prior file not found: {prior_file}")
    
    print(f"Loading prior file: {prior_file}")
    df = pd.read_csv(prior_file)
    
    # Check required columns
    required_columns = ['id', 'generate', 'real']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")
    
    print(f"Loaded {len(df)} recommendations")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Calculate accuracy metrics
    print("="*80)
    print("PRIOR RECOMMENDATION ACCURACY METRICS")
    print("="*80)
    
    total = len(df)
    correct = 0
    partial_matches = 0
    unknown_count = 0
    
    for idx, row in df.iterrows():
        generate = str(row['generate']).lower().strip()
        real = str(row['real']).lower().strip()
        
        # Count "Unknown" recommendations
        if generate == "unknown" or generate == "":
            unknown_count += 1
        
        # Exact match
        if generate == real:
            correct += 1
        else:
            # Check for partial matches (word overlap)
            generate_words = set(generate.split())
            real_words = set(real.split())
            if len(generate_words) > 0 and len(real_words) > 0:
                common_words = generate_words.intersection(real_words)
                similarity = len(common_words) / max(len(generate_words), len(real_words))
                if similarity > 0.5:  # More than 50% word overlap
                    partial_matches += 1
    
    hit_at_1 = correct / total if total > 0 else 0.0
    
    print(f"Total samples: {total}")
    print(f"Hit@1 (Exact Match): {correct} / {total} = {hit_at_1:.4f} ({hit_at_1*100:.2f}%)")
    print(f"Unknown/Empty recommendations: {unknown_count} / {total} = {unknown_count/total:.4f} ({unknown_count/total*100:.2f}%)")
    if total - correct > 0:
        print(f"Partial matches (>50% word overlap): {partial_matches} / {total - correct} = {partial_matches/(total-correct):.4f} ({partial_matches/(total-correct)*100:.2f}% of incorrect)")
    print("="*80)
    
    # Show some examples of correct and incorrect predictions
    print("\nSample Results:")
    print("-"*80)
    print("Correct predictions (first 5):")
    correct_samples = df[df['generate'].str.lower().str.strip() == df['real'].str.lower().str.strip()].head(5)
    for idx, row in correct_samples.iterrows():
        print(f"  ID {row['id']}: ✓")
        print(f"    Generate: {row['generate'][:100]}...")
        print(f"    Real:     {row['real'][:100]}...")
    
    print("\nIncorrect predictions (first 5):")
    incorrect_samples = df[df['generate'].str.lower().str.strip() != df['real'].str.lower().str.strip()].head(5)
    for idx, row in incorrect_samples.iterrows():
        print(f"  ID {row['id']}: ✗")
        print(f"    Generate: {row['generate'][:100]}...")
        print(f"    Real:     {row['real'][:100]}...")
    
    return {
        'total': total,
        'correct': correct,
        'hit_at_1': hit_at_1,
        'unknown_count': unknown_count,
        'partial_matches': partial_matches
    }


if __name__ == '__main__':
    args = get_args()
    evaluate_prior_accuracy(args.prior_file)


