from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pickle
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import time


class SrsLdaModel:
    def __init__(self, num_models, corpus, dictionary, text):
        self.limit = num_models
        self.start = 1
        self.step = 1
        self.corpus = corpus
        self.id2word = dictionary
        self.processedText = text
        self.coherence_values = []
        self.lda_model_list = []
        self.model_number = None
        self.slctd_model = None
        self.slctd_model_coh_score =None

    def create_models(self):
        print("Making models please standby")
        stop_point = self.limit + 1
        start = time.time()
        for x in range(self.start,stop_point,self.step):
            lda_model = LdaModel(corpus=self.corpus, id2word=self.id2word,num_topics=x, random_state=100,update_every=1,
                                 chunksize=100, passes=10, alpha='auto', per_word_topics=True)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=self.processedText, dictionary=self.id2word, coherence='c_v')
            self.lda_model_list.append(lda_model)
            self.coherence_values.append(coherence_model_lda.get_coherence())
        end = time.time()
        print("Model generation completed in "+str(end-start)+" seconds.")

    def plot_coherence_scores(self):
        """Returns a Plotly Figure of Coherence Scores"""
        point = None
        point in range(self.start, self.limit, self.step)
        coh_fig = plt.plot(point, self.coherence_values)
        coh_fig.xlabel('Number of Topics')
        coh_fig.ylabel('Coherence Score')
        coh_fig.title('LDA Topic Models')
        plotly_fig = mpl_to_plotly(coh_fig)
        return plotly_fig

    def select_best_model(self, slctd_model_number):
        self.model_number = slctd_model_number
        self.slctd_model = self.lda_model_list[self.model_number - 1]
        self.slctd_model_coh_score = self.coherence_values[self.model_number - 1]

    def get_selected_model_coherence_score(self):
        return self.slctd_model_coh_score

    def get_selected_lda_model(self):
        return self.slctd_model

    def get_lda_models(self):
        return self.lsa_model_list

    def get_coherence_scores(self):
        return self.coherence_values

    def get_start(self):
        return self.start

    def get_step(self):
        return self.step

    def get_limit(self):
        return self.limit

    def set_start(self, value):
        self.start = value

    def set_step(self, value):
        self.step = value

    def set_limit(self, value):
        self.limit = value

    def save_lda_model(self, filepath):
        self.slctd_model.save(filepath)

    def save_model_topic_terms(self,filepath):
        self.topic_terms = open(filepath,"w")
        for topic in range(0, len(self.slctd_model.get_topics())):
            self.topic_terms.write(self.slctd_model.print_topic(topicno=topic,topn=30))
            self.topic_terms.write("\n")
        self.topic_terms.close()
