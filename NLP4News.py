# from CS4490Z_Thesis.utils import testing_script
from utils import MyDataset,Preprocess
import gensim.corpora as corpora
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from matplotlib.patches import Rectangle
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from summarizer import Summarizer
import nltk

class LDA_Topic_Model:

    def __init__(self, df, n_comp =[4, 5, 6, 7, 8], ld = [.5, .7, .9] ):

        self.data = df

        print('Beginning vecorization')
        vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

        self.data_vectorized = vectorizer.fit_transform(Preprocess.lemmatize(df))

        print('Vecorization complete')

        search_params = {'n_components': n_comp, 'learning_decay': ld}

        # Init the Model
        lda = LatentDirichletAllocation()

        print('Beginning grid search')

        # Init Grid Search Class
        self.gs_model = GridSearchCV(lda, param_grid=search_params)

        # Do the Grid Search
        self.gs_model.fit(self.data_vectorized)
        self.best_lda_model = self.gs_model.best_params_
        print('Completed grid search')

        self.dict = corpora.Dictionary(df)
        self.corpus = [self.dict.doc2bow(text) for text in df]

        print('Constructing LDA')

        self.model = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dict,
            num_topics=self.gs_model.best_params_['n_components'], random_state=100, update_every=1, chunksize=10, passes=10, 
            alpha='symmetric', iterations=100, per_word_topics=True)
        print('Competed LDA')

        # topics = self.model.print_topics(num_words = 20)
        # for i in topics: print(i)

    def get_dominant_topics(self):

        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(self.model[self.corpus]):
            row = row_list[0] if self.model.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(self.data)
        df_topic_sents_keywords = pd.concat([sent_topics_df, contents], axis=1)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        return df_dominant_topic

    def get_topic_word_cloud(self, topic_id):

        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        topics = self.model.show_topics(formatted=False)

        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


        cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[topic_id],
                  prefer_horizontal=1.0)

        plt.figure(figsize=(10,10))
        topic_words = dict(topics[topic_id][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(topic_id), fontdict=dict(size=16))
        plt.gca().axis('off')
        plt.savefig(f'topic_{topic_id}_word_cloud.jpeg')

    def get_topic_word_dist(self, topic_id):

        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        df_dominant_topic = self.get_dominant_topics()

        plt.figure(figsize=(10,10))
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == topic_id, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        plt.hist(doc_lens, bins = 1000, color=cols[topic_id])
        plt.title(label=f'Document Word Count for Topic {topic_id}')
        plt.ylabel('Number of Documents')
        plt.savefig(f'topic_{topic_id}_word_distribution.jpeg')

    def get_sentence_colors(self, start, end):

        corp = self.corpus[start:end]
        mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
                                
        axes[0].axis('off')
        for i, ax in enumerate(axes):
            if i > 0:
                corp_cur = corp[i-1] 
                topic_percs, wordid_topics, wordid_phivalues = self.model[corp_cur]
                word_dominanttopic = [(self.model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
                ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                        fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

                # Draw Rectange
                topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
                ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                                        color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

                word_pos = 0.06
                for j, (word, topics) in enumerate(word_dominanttopic):
                    if j < 14:
                        ax.text(word_pos, 0.5, word,
                                horizontalalignment='left',
                                verticalalignment='center',
                                fontsize=16, color=mycolors[topics],
                                transform=ax.transAxes, fontweight=700)
                        word_pos += .009 * len(word)  # to move the word for the next iter
                        ax.axis('off')
                ax.text(word_pos, 0.5, '. . .',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color='black',
                        transform=ax.transAxes)       

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
        plt.tight_layout()
        plt.show() 
        plt.savefig(f'word_and_article_{start}_{end}.png')

    def get_sne_cluster(self):

        # Get topic weights
        topic_weights = []
        for i, row_list in enumerate(self.model[self.corpus]):
            topic_weights.append([w for i, w in row_list[0]])

        # Array of topic weights    
        arr = pd.DataFrame(topic_weights).fillna(0).values

        # Keep the well separated points (optional)
        arr = arr[np.amax(arr, axis=1) > 0.35]

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        # Plot the Topic Clusters using Bokeh
        output_notebook()
        n_topics = 4
        mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                    plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
        show(plot)
        plt.savefig(f't_sne_clustering_lda.png')

class BERT_topic_Model:

    def __init__(self, texts, dates):

        self.articles =  texts
        self.dates = dates
        self.model = BERTopic()
        self.topics, _ = self.model.fit_transform(self.articles)

    def get_num_topics(self):

        return len(self.model.get_topic_info())

    def get_topic_desc(self):

        return (self.model.get_topic_info())

    def show_topic_clusters(self):

        self.model.visualize_topics()

    def show_dynamic_topic_plots(self, num_topics):

        topics_over_time = self.model.topics_over_time(self.articles, self.topics, self.dates, nr_bins=20)
        self.model.visualize_topics_over_time(topics_over_time, top_n_topics=num_topics)

    def get_topic_table(self):

        self.topic_matrix = pd.DataFrame(0, index=np.arange(len(self.articles)), columns=['Topic 1','Topic 2', 'Topic 3', 'Topic 4','Topic 5'])

        for idx, i in enumerate(self.articles):

            # if (idx % 100) == 0: print(f'{idx} topics classified')
            
            topic_list = self.model.find_topics(i)[0]
            # print(topic_list)

            self.topic_matrix['Topic 1'][idx] = topic_list[0]
            self.topic_matrix['Topic 2'][idx] = topic_list[1]
            self.topic_matrix['Topic 3'][idx] = topic_list[2]
            self.topic_matrix['Topic 4'][idx] = topic_list[3]
            self.topic_matrix['Topic 5'][idx] = topic_list[4]
            # self.model.head()

        return self.topic_matrix

class VADER_sentiment_analyzer:

    def __init__(self, texts):

        nltk.download(["names","stopwords","state_union","averaged_perceptron_tagger","vader_lexicon","punkt"])

        self.articles = texts
        self.polarity =  [0] * len(texts)

        sia = SentimentIntensityAnalyzer()

        for idx, i in enumerate(texts):

            self.polarity[idx] += sia.polarity_scores(str(i))['compound']

        self.polarity_matrix = pd.DataFrame(self.polarity, columns=['Polarity'])

    def get_text_polarity(self):

        return self.polarity_matrix

class BERT_Summarization_Model:

    def __init__(self, texts):

        self.model = Summarizer()
        self.data = texts

        # abstracts = [''] * len(texts)

        # for idx, i in enumerate(texts):

        #     abstracts[idx] = self.model(i, num_sentences = 3)

        # self.summs = pd.DataFrame(abstracts, columns=['Summarizations'])

    def get_spec_text_sum(self, text_id, n_sentences):

        try:
            
            return self.model(self.data[text_id], num_sentences = n_sentences)

        except:

            return -1

class NLP_Controller:

    def __init__(self,file_name):

        self.df = MyDataset(file_name)
        # self.size = len(self.df.dataset['Heading'])

    def tm_bert(self):

        articles, dates = self.df.preprocess_for_bert()
        return BERT_topic_Model(articles, dates) 

    def tm_lda(self):

        return LDA_Topic_Model(self.df.preprocess_for_LDA())

    def sa_vader(self):

        # return VADER_sentiment_analyzer(self.df.preprocess_for_bert())
        return VADER_sentiment_analyzer(self.df.dataset.Heading)
        # print(self.df.dataset.Heading)
    
    def sum_bert(self):

        # articles, _ = self.df.preprocess_for_bert()
        articles = self.df.dataset.Article
        return BERT_Summarization_Model(articles)

    def get_dataset(self):

        return self.df

    def get_full_table(self,topic_model, polarity_model):

        self.df.dataset.reset_index(drop=True, inplace=True)

        polarity_matrix = polarity_model.get_text_polarity()
        polarity_matrix.reset_index(drop=True, inplace=True)

        topic_matrix = topic_model.get_topic_table()
        topic_matrix.reset_index(drop=True, inplace=True)
        
        self.final_data = pd.concat([self.df.dataset,polarity_matrix, topic_matrix], axis = 1)
        self.final_data['Date'] =  pd.to_datetime(self.final_data['Date'])                

        return self.final_data

    def get_topic_polarity_plots(self,topic_num, topic_model, polarity_model):

        if (topic_num < topic_model.get_num_topics() and topic_model.get_num_topics() > -1):

            final_data = self.get_full_table(topic_model, polarity_model)
            top_articles = [final_data.loc[(final_data['Topic 1'] == topic_num)],final_data.loc[(final_data['Topic 2'] == topic_num)],final_data.loc[(final_data['Topic 3'] == topic_num)],final_data.loc[(final_data['Topic 4'] == topic_num)],final_data.loc[(final_data['Topic 5'] == topic_num)]]
            articles_from_topic = pd.concat(top_articles, axis = 0)

            plt.figure(figsize=(25,10))
            plt.scatter(articles_from_topic['Date'], articles_from_topic['Polarity'])
            plt.show()
            plt.savefig(f'topic_{topic_num}_polarity_plot.png')
            plt.clf()
            plt.cla()
            plt.close()


        else: print('invalid topic number')





    


def lda_testing_script():

    df = MyDataset('Articles.csv')
    lda_model = LDA_Topic_Model(df.preprocess_for_LDA())
    lda_model.get_dominant_topics()
    lda_model.get_topic_word_cloud(2)
    lda_model.get_topic_word_dist(2)
    lda_model.get_sentence_colors(2,4)
    lda_model.get_sne_cluster()

def bert_testing_script():

    df = MyDataset('Articles.csv')
    articles, dates = df.preprocess_for_bert()
    bert_model = BERT_topic_Model(articles, dates)
    bert_model.get_num_topics()
    bert_model.get_topic_desc()
    bert_model.show_topic_clusters()
    bert_model.show_dynamic_topic_plots(10)

def vader_testing_script():

    df = MyDataset('Articles.csv')
    vader_model = VADER_sentiment_analyzer(df.preprocess_for_bert())
    vader_model.get_text_polarity()

def bert_s_testing_script():

    df = MyDataset('Articles.cvs')
    articles, _ = df.preprocess_for_bert()
    bert_s_model = BERT_Summarization_Model(articles)
    bert_s_model.get_spec_text_sum()

def main():

    # lda_testing_script()
    # bert_testing_script()
    # vader_testing_script()
    # bert_s_testing_script()

    # vader_model = NLP_Controller('Articles.csv').sa_vader()
    model = NLP_Controller('Articles.csv').tm_bert()


if __name__=="__main__":

    main()