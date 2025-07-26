#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import (T5ForConditionalGeneration,
                          AutoTokenizer,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)

model_name = "google/t5-efficient-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name)

new_tokens = [str(i) for i in range(1, 14)] + ["+", "-", "*", "/", "<no_solution>"]
num_added = tokenizer.add_tokens(new_tokens)
print("Added tokens:", num_added)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))

raw_ds = load_dataset("json",
                      data_files={"train": "train.jsonl",
                                  "validation": "val.jsonl"})

def preprocess(ex):
    inputs = "solve: " + ex["input"]
    print(inputs)
    targets = ex["output"]
    print(targets)
    model_inputs = tokenizer(inputs, max_length=20, truncation=True)
    labels = tokenizer(text_target=targets, max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = raw_ds.map(preprocess, batched=False)

args = Seq2SeqTrainingArguments(
    output_dir="./24points_rpn_nosol",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-4,
    num_train_epochs=150,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,
    logging_steps=150,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
trainer.save_model("24points_rpn_nosol")
tokenizer.save_pretrained("24points_rpn_nosol")