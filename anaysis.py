import nltk
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
stopwords_eng = stopwords.words('english')

def get_comment_text(path):
    '''
    :param path: csv path, type=str()
    :return: comment in lowercase, type=str()
    '''
    df = pd.read_csv(path)
    df['label']=df.apply(lambda row: 0 if row['toxic']==0 and row['severe_toxic']==0 and row['obscene']== 0 and row['threat']==0 and row['insult']==0 and row['identity_hate']==0 else 1, axis=1)
    toxic_df = df[df['label'] == 1]
    sent_list = []
    for data in toxic_df.comment_text:
        sent_list.append(data.lower())
    return sent_list


def clean_data(line):
    '''
    :param: line, one line in the file
    :return: a list of processed tokens
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    return_term = []
    sents = nltk.sent_tokenize(line)
    for sent in sents:
        words = nltk.word_tokenize(sent)
        for word in words:
            word = wnl.lemmatize(word).lower()
            if (word not in string.punctuation) and (word not in stopwords_eng) and (len(word) > 4):
                return_term.append(word)
    return return_term

def preprocess(sent_list):
    '''
    :param sent_list: type=list(), sentences
    :return: type=str(), all words
    '''
    terms = []
    for sent in sent_list:
        for term in clean_data(sent):
            terms.append(term)
    return " ".join(terms)

def draw_WordCloud(text):
    '''
    generate word cloud
    :param text:
    :return:
    '''
    wordcloud = WordCloud(collocations=False,
        width = 3000,
        height = 2000,
        background_color = 'black').generate(text)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('word_cloud.png')


path = "data/train.csv"
toxic_comment = get_comment_text(path)
terms = preprocess(toxic_comment)
print("terms generated finish")
draw_WordCloud(terms)