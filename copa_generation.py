#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Project       : ML4Science Project
@File          : copa_generation.py
@Author        : Yiyang Feng
@Date          : 2022/12/10 14:23
@Version       : 1.0
"""

"""
The cider evaluation metric is from https://github.com/cui-shaobo/t-conan/tree/main/models/generation/Metrics/cider. The core codes of training are referred from https://huggingface.co/docs/transformers/tasks/language_modeling
"""

import os
import math
import wandb
import torch
import argparse
import evaluate
import numpy as np
from tqdm import tqdm
from cider.cider import Cider
from dataclasses import dataclass
from typing import Optional, Union
from datasets import load_dataset, load_metric

from transformers import pipeline
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers.trainer_callback import PrinterCallback
from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, BartTokenizer
from transformers import BertLMHeadModel, RobertaForCausalLM, XLMRobertaForCausalLM, BartForConditionalGeneration
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


CONTEXT_COL = "premise"
QUESTION_COL = "question"
ANSWER_1_COL = "choice1"
ANSWER_2_COL = "choice2"
LABEL_COL = "label"


def preprocess_for_lm(examples, tokenizer):
    question_headers = examples[QUESTION_COL]
    first_sentences = [
        f"{examples[CONTEXT_COL][i]} What was the CAUSE of this?" if header == "cause" else\
        f"{examples[CONTEXT_COL][i]} What was the EFFECT of this?"\
            for i, header in enumerate(question_headers)
    ]
    labels = examples[LABEL_COL]
    
    second_sentences = [
        f"{examples[ANSWER_1_COL][i]}" if label == 0 else\
        f"{examples[ANSWER_2_COL][i]}"\
            for i, label in enumerate(labels)
    ]

    sentences = [
        f"{first_sentences[i]} {second_sentences[i]}" for i, _ in enumerate(first_sentences)
    ]

    output = tokenizer(sentences, truncation=True, padding=True)
    output["labels"] = output["input_ids"].copy()

    return output


def preprocess_for_seq2seq(examples, tokenizer):
    question_headers = examples[QUESTION_COL]
    first_sentences = [
        f"{examples[CONTEXT_COL][i]} What was the CAUSE of this?" if header == "cause" else\
        f"{examples[CONTEXT_COL][i]} What was the EFFECT of this?"\
            for i, header in enumerate(question_headers)
    ]
    labels = examples[LABEL_COL]
    
    second_sentences = [
        f"{examples[ANSWER_1_COL][i]}" if label == 0 else\
        f"{examples[ANSWER_2_COL][i]}"\
            for i, label in enumerate(labels)
    ]

    model_inputs = tokenizer(first_sentences, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(second_sentences, max_length=32, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(predictions, references, results={}):
    """ compute BLEU-1 to BLEU-4, METEOR, ROUGE-L, CIDEr
    """
    
    # load evaluation metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    cider = Cider()

    # compute bleu 1-4
    for i in range(1, 5):
        results[f"bleu_{i}"] = bleu.compute(predictions=predictions, references=references, max_order=i)["bleu"]

    # compute meteor
    results["meteor"] = meteor.compute(predictions=predictions, references=sum(references, []))["meteor"]

    # compute rouge
    results["rouge_L"] = rouge.compute(predictions=predictions, references=references)["rougeL"]

    # compute cider
    results["cider"] = cider.compute(predictions, references)[0]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert", help="Model choice: bert, roberta, xlmroberta, or albert")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Model output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
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

    if args.model == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertLMHeadModel.from_pretrained("bert-base-uncased", is_decoder=True)
    elif args.model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer.pad_token = tokenizer.eos_token
        model = RobertaForCausalLM.from_pretrained("roberta-base", is_decoder=True)
    elif args.model == "xlmroberta":
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        tokenizer.pad_token = tokenizer.eos_token
        model = XLMRobertaForCausalLM.from_pretrained("xlm-roberta-base", is_decoder=True)
    elif args.model == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        tokenizer.pad_token = tokenizer.eos_token
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    else:
        print("Model Not Supported")
        raise ModuleNotFoundError

    # Load COPA dataset
    copa = load_dataset("super_glue", "copa")

    # Data preprocessing
    if args.model == "bart":
        tokenized_copa = copa.map(
            lambda f: preprocess_for_seq2seq(f, tokenizer),
            batched=True,
            remove_columns=copa["train"].column_names
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        tokenized_copa = copa.map(
            lambda f: preprocess_for_lm(f, tokenizer),
            batched=True,
            remove_columns=copa["train"].column_names
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
        data_collator=data_collator,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    # Get predictions from validation set
    predictions = []
    references = []

    for idx, example in tqdm(enumerate(copa["validation"])):
        example = copa["validation"][idx]
        if example["question"] == "cause":
            prompt = f"{example[CONTEXT_COL]} What was the CAUSE of this?"
        elif example["question"] == "effect":
            prompt = f"{example[CONTEXT_COL]} What was the EFFECT of this?"
        label = example[LABEL_COL]
        truth = example[ANSWER_1_COL] if label == 0 else example[ANSWER_2_COL]
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        if args.model == "bart":
            outputs = model.generate(inputs, max_new_tokens=100, do_sample=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pos_e = answer.find(".")
            answer = answer[:pos_e+1]
        else:
            outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
            answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt)+1:]
            pos_e = answer.find(".")
            pos_s = answer.find("?")
            if args.model == "bert":
                pass
            elif args.model == "roberta" or args.model == "xlmroberta":
                answer = answer[pos_s+1:pos_e+1]
            else:
                answer = answer[:pos_e+1]

        predictions.append(answer)
        references.append([truth])

    # compute metrics
    results = {}
    results["preplexity"] = math.exp(trainer.evaluate()["eval_loss"])
    results = compute_metrics(predictions, references, results)

    # Predicting test dataset (no label)
    if args.do_test:
        predictions = []
        for idx, example in enumerate(test_dataset):
            if example["question"] == "cause":
                prompt = f"{examples[CONTEXT_COL]} What was the CAUSE of this?"
            elif example["question"] == "effect":
                prompt = f"{examples[CONTEXT_COL]} What was the EFFECT of this?"
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            if args.model == "bart":
                outputs = model.generate(inputs, max_new_tokens=100, do_sample=True)
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt)+1:]
                pos_e = answer.find(".")
                pos_s = answer.find("?")
                if args.model == "bert":
                    pass
                elif args.model == "roberta" or args.model == "xlmroberta":
                    answer = answer[pos_s+1:pos_e+1]
                else:
                    answer = answer[:pos_e+1]
            predictions.append(answer)

        output_predict_file = os.path.join("./", args.predict_dir)
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = predictions[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()