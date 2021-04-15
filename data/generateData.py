import re
import time
import pandas as pd
import string
def text_preprocessing(text):
    text = re.sub(r'https?://\S+|www\.\S+', r'', text) # remove urls
    text = re.sub(r'(@.*?)[\s]', ' ', text) # remove @username
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'[0-9]+' , '' ,text) # remove numbers
    text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
    text = re.sub(r'&amp;', '&', text) # remove special html terms
    text = re.sub(r'<[^>]+>]', '', text) # remove html tags
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("#" , " ")

    ##### expand contractions ######
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"ain\'t","is not", text)
    text = re.sub(r"shan\'t", "shall not", text)
    text = re.sub(r"let\'s", "let us", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)

    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    encoded_string = text.encode("ascii", "ignore") # remove non-english words
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
    i=0
    for index in df.index:
        i+=1
        if i % 1000 == 0:
            print(i)
        raw = df.loc[index,'comment_text']
        df.loc[index, 'comment_text'] = text_preprocessing(raw)

    df.to_csv("cleaned.csv", index=0)
    tick2 = time.time()
    print("time consumption:", tick2-tick1)

TRAIN = "train.csv"
TEST = "test.csv"
RAW = "raw_data.csv"
# generate_new_label(TRAIN)
str = "中华 raw data"
print(text_preprocessing(str))

