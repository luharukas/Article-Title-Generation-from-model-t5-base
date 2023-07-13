import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from transformers import AutoModelForSeq2SeqLM,T5TokenizerFast
from transformers import AutoConfig

import configuration as cfg

submission = pd.read_csv('train.csv')
model = AutoModelForSeq2SeqLM.from_pretrained(cfg.PATH_TO_MODEL)
tokenizer = T5TokenizerFast.from_pretrained(cfg.PATH_TO_MODEL)
config = AutoConfig.from_pretrained(cfg.PATH_TO_MODEL)
submission=submission.loc[:20,:]

def predict(texts):
    # write code to output a list of title for each text input to the predict method
    corpus_data=texts.tolist()
    input_data = tokenizer(corpus_data, max_length=600, return_tensors='pt',truncation=True, padding=True,)
    title_idx = model.generate(
    input_data['input_ids'],
    temperature=cfg.TEMPERATURE,
    max_length=cfg.MAX_GEN_LEN,
    num_beams=cfg.NUM_BEAM,
    early_stopping=True)
    prediction=[]
    for idx in range(len(corpus_data)):
      title = tokenizer.decode(title_idx[idx], skip_special_tokens=True, clean_up_tokenization_spaces=False,)
      prediction.append(title)
    score=evaluate(prediction,submission['title'].tolist())
    print(score)
    return prediction

def test_model():
    pred = predict(submission['text'])
    submission['predicted_title'] = pred
    submission.to_csv('submission.csv',index=False)


def evaluate(model_output,actual_titles):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = list()
    for output,actual in zip(model_output,actual_titles):
        s = scorer.score(output,actual)
        scores.append(s['rouge1'].fmeasure)

    print('Evaluation result',np.mean(scores))
    return scores




if __name__=="__main__":
    #write model loading code here
    test_model()
