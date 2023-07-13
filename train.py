import pandas as pd
import configuration
from function import(
    dataset_processing,
    corpus_preprocessing,
    data_preparation,
    preprocessing,
    training_args,
    load_model,
    tokenizer,
    compute_metrics
)
from transformers import Seq2SeqTrainer,DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")


def train():
    #write training code here
    data=pd.read_csv("train.csv",sep=",",encoding='utf-8')
    INPUT_DATA, OUTPUT_DATA =dataset_processing(data)
    PROCESSED_INPUT=INPUT_DATA.apply(corpus_preprocessing,use_as="input")
    PROCESSED_OUTPUT=OUTPUT_DATA.apply(corpus_preprocessing,use_as="output")
    final_data=pd.DataFrame({"text":PROCESSED_INPUT,"title":PROCESSED_OUTPUT})
    final_data.to_csv("processed_train.csv",sep=",")
    data=data_preparation("processed_train.csv")
    data_processed=data.map(preprocessing,batched=True,remove_columns=['text', 'title'],)
    model=load_model()
    collator=DataCollatorForSeq2Seq(tokenizer,model=model)
    trainer=Seq2SeqTrainer(model=model,args=training_args,train_dataset=data_processed['train'],eval_dataset=data_processed['val'],data_collator=collator,tokenizer=tokenizer,compute_metrics=compute_metrics)
    print("Passed till here")
    print("Training Started")
    trainer.train()

    
if __name__=="__main__":
    print('Running training script')
    train()