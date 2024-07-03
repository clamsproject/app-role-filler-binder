"""
Prepare curated annotation dataset for ner token classification task.
The input file is expected to be a csv with the following columns: ['guid', 'scene_label', 'cleaned_text', 'labels']

An exception will be raised if for any row, the number of tokens in 'cleaned_text' does not match the number of tags
in 'labels'.

Outputs 3 json files for train/val/test data partitions.

Usage: python3 prepare_data.py --data path/to/data.csv
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def get_tokens(cleaned_text: str):
    """
    Get tokens from preprocessed OCR text.
    """
    return cleaned_text.split()


def get_labels(silver_annotation: str):
    """
    Extract BIO tags from a sequence of spans in the form `token@tag:index`
    """
    mappings = {'BR': 'B-ROLE', 'IR': 'I-ROLE', 'BF': 'B-FILL', 'IF': 'I-FILL'}
    labels = []
    for labeled_tok in silver_annotation.split():
        tok, label = labeled_tok.split('@')
        if ':' in label:
            label = label.split(':')[0]
            label = mappings[label]
        labels.append(label)
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to annotation csv file')
    args = parser.parse_args()

    original_df = pd.read_csv(args.data)
    tokens = original_df['scene_label'] + " " + original_df['cleaned_text']
    tokens = tokens.map(get_tokens)
    labels = original_df['labels'].map(get_labels).tolist()
    labels = [['O'] + label_seq for label_seq in labels]  # first element will be the scene classification
    for idx, tok_seq in enumerate(tokens):
        lab_seq = labels[idx]
        # require the token and tag seqs to be the same length
        assert len(lab_seq) == len(tok_seq), \
            (f"Length of tokens ({len(tok_seq)}) does not match length of labels ({len(lab_seq)}):\n"
             f"({tok_seq}, {lab_seq})")
    output_df = pd.DataFrame(data={"tokens": tokens, "labels": labels})
    train, val = train_test_split(output_df, test_size=0.2, random_state=42)
    train.to_json('../model_in_data/rfb_train.json', orient='records', lines=True)
    val, test = train_test_split(val, test_size=0.5, random_state=42)
    val.to_json('../model_in_data/rfb_val.json', orient='records', lines=True)
    test.to_json('../model_in_data/rfb_test.json', orient='records', lines=True)
