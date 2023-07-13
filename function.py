import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import numpy as np
nltk.download('punkt')
import re
import string
import rouge_score
import contractions
from datasets import load_dataset,load_from_disk
from transformers import pipeline, T5TokenizerFast
import configuration as cfg
from transformers import Seq2SeqTrainingArguments,AutoModelForSeq2SeqLM,Seq2SeqTrainer


# Preprocessing Configurations
punctuation=string.punctuation.replace('.',"")
mapper=contractions.contractions_dict

def dataset_processing(data):
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data=data.reset_index(drop=True)
    return data['text'],data['title']

def remove_url(corpus):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',corpus)

def remove_mention(corpus):
    url = re.compile(r'@\S*')
    return url.sub(r'',corpus)

def remove_punctuation(corpus):
    table=str.maketrans('','',punctuation)
    return corpus.translate(table)

def unicode_to_ascii(corpus):
    corpus = corpus.encode('utf8').decode('ascii', 'ignore')
    return corpus

def replace_mapping(text):
    for p in mapper:
        text = text.replace(p, mapper[p])
    return text


def corpus_preprocessing(corpus,use_as=None):
    if use_as=="input":
        corpus="".join(corpus.split("-")[1:])
    corpus=remove_url(corpus)
    corpus=remove_mention(corpus)
    corpus=remove_punctuation(corpus)
    corpus=unicode_to_ascii(corpus)
    corpus=replace_mapping(corpus)
    return " ".join(word_tokenize(corpus.lower()))


def data_preparation(datafile):
    full_dataset = load_dataset("csv", data_files=datafile)
    cols_to_remove = list(full_dataset['train'].features.keys())
    cols_to_remove.remove("title")
    cols_to_remove.remove("text")
    data=full_dataset.remove_columns(cols_to_remove)
    dataset = data['train'].train_test_split(test_size=0.1)
    test_val = dataset['test'].train_test_split(test_size=0.5)
    dataset['val'] = test_val['train']
    dataset['test'] = test_val['test']
    dataset.save_to_disk("final_data")
    dataset = load_from_disk("final_data")
    return dataset

tokenizer = T5TokenizerFast.from_pretrained("t5-base",model_max_length=cfg.MAX_TEXT_LENGTH)
def preprocessing(data):
    model_inputs = tokenizer(data['text'], max_length=cfg.MAX_TEXT_LENGTH, padding=True, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(data['title'], max_length=cfg.MAX_TITLE_LENGTH, padding=True, truncation=True)
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

training_args = Seq2SeqTrainingArguments(
    output_dir="model-t5-base",
    evaluation_strategy="steps",
    eval_steps=cfg.EVAL_EVERY,
    learning_rate=cfg.LEARNING_RATE,
    per_device_train_batch_size=cfg.BATCH_SIZE,
    per_device_eval_batch_size=cfg.BATCH_SIZE,
    weight_decay=cfg.WEIGHT_DECAY,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=cfg.EPOCHS,
    predict_with_generate=True,
    logging_steps=cfg.LOG_EVERY,
    group_by_length=True,
    lr_scheduler_type=cfg.LR_SCHEDULAR_TYPE,
    resume_from_checkpoint=True,
)

def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    return model


from datasets import load_metric
metric = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}







