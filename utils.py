# from _typeshed import Self
import pandas as pd
import re
import gensim
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import nltk
import spacy

class MyDataset:

    def __init__(self, fn):

        self.filename = fn
        self.dataset = pd.read_csv(fn, encoding="ISO-8859-1")

    def preprocess_for_LDA(self):

        data = Preprocess.text_to_list(self.dataset.Article)
        data_words = list(Preprocess.remove_special_chars(data))
        clean_data = Preprocess.remove_stopwords(data_words)

        return clean_data

    def preprocess_for_bert(self):

        # clean_data = (Preprocess.remove_special_chars(self.dataset.Article))
        self.dataset.Article =self.dataset.apply(lambda row: re.sub(r"http\S+", "", row.Article).lower(), 1)
        self.dataset.Article = self.dataset.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.Article.split())), 1)
        self.dataset.Article = self.dataset.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.Article).split()), 1)
        clean_data = self.dataset.Article.to_list()
        dates = list(self.dataset.Date)
        return clean_data, dates

    def preprocess_for_VADER(self):

        return Preprocess.remove_special_chars(self.dataset.Heading)

    def show(self):

        print(self.dataset.head())

class Preprocess:

    def text_to_list(texts):

        return texts.values.tolist()

    def remove_special_chars(Article):

        for s in Article:
            s = re.sub('\S*@\S*\s?', '', s)  # remove emails
            s = re.sub('\s+', ' ', s)  # remove newline chars
            s = re.sub("\'", "", s)  # remove single quotes
            s = gensim.utils.simple_preprocess(str(s), deacc=True) 
            yield(s)

    def remove_stopwords(Articles):

        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']

        bigram = gensim.models.Phrases(Articles, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[Articles], threshold=100)  
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in Articles]
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []
        # run this if you get an error here: python -m spacy download en
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        
        
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
        return texts_out
    
    def lemmatize(Articles):

        allowed_postags=['NOUN','ADJ','VERB','ADV']
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        texts_out = []

        for s in Articles:
            doc = nlp(" ".join(s))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        return texts_out

def testing_script():

    # Testing Loading Data
    x = MyDataset('Articles.csv')
    x.show()

    # Testing lext_to_list
    data = Preprocess.text_to_list(x.dataset.Article)
    print(data[:1])

    # Testing data_words
    data_words = list(Preprocess.remove_special_chars(data))
    print(data_words[:1])

    # Testing remove_stopwords
    clean_data = Preprocess.remove_stopwords(data_words)
    print(clean_data[:1])

    # Testing lemmatize
    data_lemmatized = Preprocess.lemmatize(data_words)
    print(data_lemmatized[:1])

def main():

    testing_script()

if __name__=="__main__":

    main()