import argparse
import inspect
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "mt5_segmented_v2" / "train.jsonl"
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "mt5_segmented_v2" / "test.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mt5-small-segmented-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 on segmented_v2 samples.")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test_file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="google/mt5-small")
    parser.add_argument("--max_source_length", type=int, default=768)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_eval_samples", type=int)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mixed precision. By default it follows CUDA availability.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class TokenizedRowsDataset:
    def __init__(self, features):
        self.features = features

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return len(self.features)


def tokenize_dataset(rows, tokenizer, max_source_length, max_target_length):
    features = []
    for row in tqdm(rows, desc="Tokenizing"):
        features.append(
            tokenizer(
            f"summarize: {row['source']}",
            text_target=row["target"],
            max_length=max_source_length,
            max_target_length=max_target_length,
            truncation=True,
        )
        )
    return TokenizedRowsDataset(features)


def use_fp16(args, cuda_available):
    return cuda_available if args.fp16 is None else args.fp16


def build_training_args(args, fp16):
    kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "fp16": fp16,
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "logging_steps": 10,
        "save_total_limit": 2,
        "report_to": [],
        "seed": args.seed,
    }
    training_arg_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "eval_strategy" in training_arg_params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return Seq2SeqTrainingArguments(**kwargs)


def generate_test_rows(model, tokenizer, rows, args):
    device = next(model.parameters()).device
    model.eval()
    output_rows = []

    for row in tqdm(rows, desc="Generating test summaries"):
        encoded = tokenizer(
            f"summarize: {row['source']}",
            max_length=args.max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_target_length,
            )
        output_rows.append(
            {
                "id": row["id"],
                "corpus": row["corpus"],
                "article_path": row["article_path"],
                "chunk_ids": row["chunk_ids"],
                "source": row["source"],
                "gold_target": row["target"],
                "generated": tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                ),
            }
        )
    return output_rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    train_rows = read_jsonl(args.train_file, args.max_train_samples)
    test_rows = read_jsonl(args.test_file, args.max_eval_samples)
    if not train_rows or not test_rows:
        raise ValueError("Train and test files must both contain at least one sample.")

    cuda_available = torch.cuda.is_available()
    fp16 = use_fp16(args, cuda_available)
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {cuda_available}")
    print(f"gpu name: {torch.cuda.get_device_name(0) if cuda_available else None}")
    print(f"fp16: {fp16}")
    print(f"train samples: {len(train_rows)}")
    print(f"test samples: {len(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenized_train = tokenize_dataset(
        train_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )
    tokenized_test = tokenize_dataset(
        test_rows,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(args, fp16),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    generations = generate_test_rows(trainer.model, tokenizer, test_rows, args)
    generations_path = args.output_dir / "test_generations.jsonl"
    write_jsonl(generations_path, generations)
    print(f"saved generations: {generations_path}")


if __name__ == "__main__":
    main()
