import argparse
from copy import deepcopy

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from evaluate import load
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

## Enable cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Disable wandb
import os
os.environ["WANDB_DISABLED"] = "true"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="microsoft/unixcoder-base-nine")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--continue_on_semi_supervisioned_training", type=bool, default=False)
    return parser.parse_args()


metric = load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


def main():
    args = get_args()
    set_seed(args.seed)

    ## Checking if is a retraining from Semi-Supervisioned Training
    origin_dataset_file_path = '../../output.parquet'
    if args.continue_on_semi_supervisioned_training == True:
        import pyarrow.parquet as pq
        semi_supervisioned_training = '../crawler/crawled_data.parquet'
        files = [origin_dataset_file_path]

        origin_dataset_file_path = 'new_output.parquet'
        schema = pq.ParquetFile(files[0]).schema_arrow
        with pq.ParquetWriter(origin_dataset_file_path, schema=schema) as writer:
            for file in files:
                writer.write_table(pq.read_table(file, schema=schema))

            for new_file in [semi_supervisioned_training]:
                ## To Avoid retraining with big schema we will limit new rows
                pf = pq.read_table(new_file, schema=schema).slice(length=2000)
                writer.write_table(pf)

    dataset = load_dataset('parquet', data_files=origin_dataset_file_path, split="train")
    train_test = dataset.train_test_split(test_size=0.1)
    test_validation = train_test["test"].train_test_split(test_size=0.5)
    train_test_validation = DatasetDict(
        {
            "train": train_test["train"],
            "test": test_validation["train"],
            "valid": test_validation["test"],
        }
    )

    num_labels = 12
    all_labels = train_test_validation["train"]["category"]
    train_labels = list(set(all_labels))
    labels = ClassLabel(num_classes=num_labels, names=train_labels, id="design_pattern")
    ## TODO: save in a json/csv file
    id2label = {}
    for t in train_labels:
        id2label[labels.str2int(t)] = t
        print("Labels Mapping: ", t, " - ", labels.str2int(t))

    label2id = {val: key for key, val in id2label.items()}

    import json
    with open('labels.json', 'w') as f:
        json.dump(label2id, f)

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id).to(device)
    model.config.pad_token_id = model.config.eos_token_id

    if args.freeze:
        for param in model.roberta.parameters():
            param.requires_grad = False

    def tokenize(example):
        inputs = tokenizer(example["context"], truncation=True, return_tensors='pt', padding='max_length', max_length=1024).to(device)
        label = labels.str2int(example["category"])
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": label,
        }

    tokenized_datasets = train_test_validation.map(
        tokenize,
        batched=True,
        remove_columns=train_test_validation["train"].column_names,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        label_names=["labels"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # accuracy
        use_cpu=False,
        run_name="category-design-patterns",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    print("Show evaluation...")
    trainer.evaluate()

    print("Saving model")
    trainer.save_model('./design-patterns-model')

    ## Use test_dataset and plot graph of categories
    ## Test predictions
    print('Number of test dataset', len(tokenized_datasets["test"]))
    raw_pred, _, _ = trainer.predict(tokenized_datasets["test"])

    y_pred = np.argmax(raw_pred, axis=1)
    print(y_pred[0])


if __name__ == "__main__":
    main()