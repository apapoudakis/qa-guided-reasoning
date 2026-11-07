import argparse
import os
from pathlib import Path

import pandas as pd
from nltk.tokenize import word_tokenize


def filter_corpus(dataset_path, tokens_threshold, output_path, task="description"):
    os.makedirs(output_path, exist_ok=True)
    data = pd.read_json(dataset_path, lines=True)

    data = data[data[task].apply(lambda x: len(word_tokenize(x)) > tokens_threshold)]
    data_name = Path(dataset_path).stem
    data.to_json(
        os.path.join(output_path, data_name + f"-filtered-{tokens_threshold}.jsonl"),
        orient="records",
        lines=True,
    )


def split_corpus(dataset_path, split_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    train = pd.read_csv(os.path.join(split_path, "train.tsv"), sep="\t")
    val = pd.read_csv(os.path.join(split_path, "val.tsv"), sep="\t")
    test = pd.read_csv(os.path.join(split_path, "test.tsv"), sep="\t")
    full_corpus = pd.read_json(dataset_path, lines=True)

    train_split = full_corpus[full_corpus["id"].isin(train["BookId"].unique())]
    train_split.to_json(
        os.path.join(save_path, "train.jsonl"), orient="records", lines=True
    )

    val_split = full_corpus[full_corpus["id"].isin(val["BookId"].unique())]
    val_split.to_json(
        os.path.join(save_path, "val.jsonl"), orient="records", lines=True
    )

    test_split = full_corpus[full_corpus["id"].isin(test["BookId"].unique())]
    test_split.to_json(
        os.path.join(save_path, "test.jsonl"), orient="records", lines=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter corpus records or apply BookWorm splits.")
    parser.add_argument(
        "--action", choices=["filter", "split"], required=True
    )
    parser.add_argument("--data_path", "--dataset_path", required=True)
    parser.add_argument("--save_path")
    parser.add_argument("--split_path")
    parser.add_argument("--task", default="description")
    parser.add_argument("--tokens_threshold", type=int)
    parser.add_argument("--output_path")
    args = parser.parse_args()

    if args.action == "filter":
        if args.tokens_threshold is None or args.output_path is None:
            parser.error(
                "--tokens_threshold and --output_path are required for filtering."
            )
        filter_corpus(
            args.data_path, args.tokens_threshold, args.output_path, args.task
        )
    elif args.action == "split":
        if args.split_path is None or args.save_path is None:
            parser.error(
                "--split_path and --save_path are required for BookWorm splitting."
            )
        split_corpus(args.data_path, args.split_path, args.save_path)
