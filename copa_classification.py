#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Project       : ML4Science Project
@File          : copa_classification.py
@Author        : Yiyang Feng
@Date          : 2022/12/10 11:14
@Version       : 1.0
"""

"""
The core codes of this file i.e., preprocess_function, compute_metrics, DataCollatorForMultipleChoice are from https://huggingface.co/docs/transformers/tasks/multiple_choice
"""

import os
import wandb
import torch
import argparse
import evaluate
import numpy as np
from random import randint
from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import PrinterCallback
from transformers import AutoTokenizer, RobertaTokenizer, XLMRobertaTokenizer, AlbertTokenizer
from transformers import AutoModelForMultipleChoice, RobertaForMultipleChoice, XLMRobertaForMultipleChoice, AlbertForMultipleChoice
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


CONTEXT_COL = "premise"
QUESTION_COL = "question"
ANSWER_1_COL = "choice1"
ANSWER_2_COL = "choice2"


def preprocess_function(examples, tokenizer):
    """
    The preprocessing function needs to:
    1. Make two copies of the CONTEXT_COL field and combine each of them with QUESTION_COL to recreate how a sentence starts.
    2. Combine QUESTION_COL with each of the two possible choices.
    3. Flatten these two lists so you can tokenize them, and then unflatten them afterward so each example has a corresponding input_ids, attention_mask, and labels field.
    """

    question_headers = examples[QUESTION_COL]
    first_sentences = [
        [f"{examples[CONTEXT_COL][i]} What was the CAUSE of this?"]*2 if header == "cause" else\
        [f"{examples[CONTEXT_COL][i]} What was the EFFECT of this?"]*2\
            for i, header in enumerate(question_headers)
    ]
    first_sentences = sum(first_sentences, [])
    
    second_sentences = [
        [examples[end][i] for end in [ANSWER_1_COL, ANSWER_2_COL]] for i, header in enumerate(question_headers)
    ]
    second_sentences = sum(second_sentences, [])
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}


def compute_metrics(eval_pred, accuracy):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", help="Model choice: bert, roberta, xlmroberta, or albert")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Model output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--run_time", type=int, default=5, help="Number of experiments")
    parser.add_argument("--wandb_logging", action="store_true", default=False, help="Whether to use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="machine-learning-copa")
    parser.add_argument("--wandb_entity", type=str, default="yiyang-feng")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name for identifying the experiment")
    parser.add_argument("--do_test", action="store_true", default=False, help="Whether to predict test dataset")
    parser.add_argument("--predict_dir", type=str, default="predictions.txt", help="Prediction generation directory")

    args = parser.parse_args()

    train_dataset = None
    val_dataset = None
    test_dataset = None

    run_time = args.run_time
    a, p, r, f = [], [], [], []

    for i in range(args.run_time):
        if args.model == "bert":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
        elif args.model == "roberta":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForMultipleChoice.from_pretrained("roberta-base")
        elif args.model == "xlmroberta":
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
            model = XLMRobertaForMultipleChoice.from_pretrained("xlm-roberta-base")
        elif args.model == "albert":
            tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
            model = AlbertForMultipleChoice.from_pretrained("albert-large-v2")
        elif args.model == "random":
            pass
        else:
            print("Model Not Supported")
            raise ModuleNotFoundError

        # Load COPA dataset
        copa = load_dataset("super_glue", "copa")

        # Load evaluation metrics
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        # compute random guess performance
        if args.model == "random":
            y_true = copa["validation"]["label"]
            a, p, r, f = [], [], [], []

            for run_time in range(args.run_time):
                y_pred = [randint(0, 1) for i in range(len(y_true))]

                a.append(accuracy.compute(predictions = y_pred, references = y_true)["accuracy"])
                p.append(precision.compute(predictions = y_pred, references = y_true, average="macro")["precision"])
                r.append(recall.compute(predictions = y_pred, references = y_true, average="macro")["recall"])
                f.append(f1.compute(predictions = y_pred, references = y_true, average="macro")["f1"])

            print(np.array(a).mean(), np.array(a).std())
            print(np.array(p).mean(), np.array(p).std())
            print(np.array(r).mean(), np.array(r).std())
            print(np.array(f).mean(), np.array(f).std())

            exit(0)

        # Data preprocessing
        tokenized_copa = copa.map(lambda f: preprocess_function(f, tokenizer), batched=True)
        train_dataset = tokenized_copa["train"]
        val_dataset = tokenized_copa["validation"]
        test_dataset = tokenized_copa["test"]

        # Load training arguments
        if args.wandb_logging:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity)
            wandb.run.name = args.wandb_run_name

            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="epoch",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                weight_decay=0.01,
                report_to="wandb",
                load_best_model_at_end = True,
                metric_for_best_model = "accuracy",
                save_strategy="epoch",
                save_total_limit=2,
            )
        else:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="epoch",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                weight_decay=0.01,
                load_best_model_at_end = True,
                metric_for_best_model = "accuracy",
                save_strategy="epoch",
                save_total_limit=2,
            )

        # Train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
            compute_metrics = lambda f: compute_metrics(f, accuracy),
        )
        trainer.remove_callback(PrinterCallback)
        trainer.train()

        # Get predictions from validation set
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")

        # compute accuracy, precision, recall, and f1
        # run multiple times of training and calculate the mean and standard error later
        a.append(accuracy.compute(predictions = predictions.argmax(axis=1), references = labels)["accuracy"])
        p.append(precision.compute(predictions = predictions.argmax(axis=1), references = labels, average="macro")["precision"])
        r.append(recall.compute(predictions = predictions.argmax(axis=1), references = labels, average="macro")["recall"])
        f.append(f1.compute(predictions = predictions.argmax(axis=1), references = labels, average="macro")["f1"])
    
    print(np.array(a).mean(), np.array(a).std())
    print(np.array(p).mean(), np.array(p).std())
    print(np.array(r).mean(), np.array(r).std())
    print(np.array(f).mean(), np.array(f).std())

    # Predicting test dataset (no label)
    if args.do_test:
        predictions = []
        for idx, example in enumerate(test_dataset):
            if example["question"] == "cause":
                prompt = f"{examples[CONTEXT_COL]} What was the CAUSE of this?"
            elif example["question"] == "effect":
                prompt = f"{examples[CONTEXT_COL]} What was the EFFECT of this?"
            choice1 = example[ANSWER_1_COL]
            choice2 = example[ANSWER_2_COL]
            inputs = tokenizer([[prompt, choice1], [prompt, choice2]], return_tensors="pt", padding=True)
            outputs = model(**{k: v.unsqueeze(0).to("cuda") for k, v in inputs.items()})
            logits = outputs.logits
            predicted_class = logits.argmax().item()
            predictions.append(predicted_class)

        output_predict_file = os.path.join("./", args.predict_dir)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = predictions[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()