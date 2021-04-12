import re
import time
import pandas as pd

def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'[0-9]+' , '' ,text)
    text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("#" , " ")
    encoded_string = text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    return decode_string


def generate_new_label(path):
    df = pd.read_csv(path)
    df['label']=df.apply(lambda row: 0 if row['toxic']==0 and row['severe_toxic']==0 and row['obscene']== 0 and row['threat']==0 and row['insult']==0 and row['identity_hate']==0 else 1, axis=1)
    df2 = df.drop(['id', 'toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)
    print(df2.head())
    df2.to_csv("raw_data.csv",index=0)

def generate_cleaned_data(path):
    df = pd.read_csv(path)
    tick1 = time.time()
    for index in df.index:
        raw = df.loc[index,'comment_text']
        df.loc[index, 'comment_text'] = text_preprocessing(raw)

    df.to_csv("cleaned.csv", index=0)
    tick2 = time.time()
    print("time consumption:", tick2-tick1)

TRAIN = "train.csv"
TEST = "test.csv"
RAW = "raw_data.csv"
# generate_new_label(TRAIN)

generate_cleaned_data(RAW)