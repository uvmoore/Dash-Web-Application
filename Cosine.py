import copy
import gensim
import pandas as pd
import gensim.corpora as corpora


class CosineSimilarity:
    def __init__(self):
        self.srs_lda_model = None
        self.lda_capec_vector = []
        self.lda_srs_vector = []

        self.srs_lsa_model = None
        self.lsa_srs_vector = []
        self.lsa_capec_vector = []

    def init_lda_data(self,lda_model):
        """Method to load in the LDA Topic Model and associated data"""
        self.srs_lda_model = lda_model
        self.srs_lda_num_topics = len(lda_model.get_topics())
        for topic in range(0, self.srs_lda_num_topics):
            self.lda_srs_vector.append(self.srs_lda_model.get_topic_terms(topicid=topic, topn=30))
        self._create_capec_vectors_lda()

    def _get_CAPEC_IDs(self,sorted_results):
        CAPEC_IDs = pd.read_csv("resources/Comprehensive CAPEC Dictionary.csv", usecols=["ID"])
        list_with_capec_ids = []
        for pattern in sorted_results:
            topic_list = []
            for value in pattern:
                index = value[0]-1
                new_tuple = tuple((CAPEC_IDs.iloc[index].values[0],value[1]))
                topic_list.append(new_tuple)
            list_with_capec_ids.append(copy.deepcopy(topic_list))
        return list_with_capec_ids

    def get_srs_lda_vector(self):
        return self.lda_srs_vector

    def get_capec_lda_vector(self):
        return self.lda_capec_vector

    def _create_capec_vectors_lda(self):
        capec_vec = []
        capec_num = 1
        for x in range(1,545):
            capec_lda = gensim.models.LdaModel.load("CAPEC/LDA/capec_lda_" + str(capec_num))
            for topic_number in range(0,4):
                capec_vec.extend(capec_lda.get_topic_terms(topicid=topic_number,topn=10))
            self.lda_capec_vector.append(capec_vec[:])
            capec_vec.clear()
            capec_num+=1
        return

    def lda_calculate_cos_sim(self):
        """Calculates cosine similarity and sorts the results"""
        self.lda_cos_result_list = []
        lda_results = []
        lda_cos_result = {}
        for x in range(0, self.srs_lda_num_topics):
            for y in range(0, 544):
                lda_cos_result.update({y: gensim.matutils.cossim(self.lda_srs_vector[x], self.lda_capec_vector[y])})
            lda_results.append(copy.deepcopy(lda_cos_result))
            lda_cos_result.clear()
       #SORTS COSINE VALUES
        for x in range(0, self.srs_lda_num_topics):
            self.lda_cos_result_list.append(sorted(lda_results[x].items(), key=lambda x: x[1], reverse=True))
        self.lda_cos_result_list = self._get_CAPEC_IDs(self.lda_cos_result_list)
        return

    def get_lda_cosine_results(self):
        return self.lda_cos_result_list

    def save_lda_cos_results(self,filepath):
        pandas_frame_lda = pd.DataFrame(self.lda_cos_result_list)
        pandas_frame_lda = pandas_frame_lda.transpose()
        for x in range(0, self.srs_lda_num_topics):
            pandas_frame_lda.rename(columns={x: "Topic-"+ str(x+1)}, inplace=True)
        print(pandas_frame_lda)
        pandas_frame_lda.to_csv(filepath,encoding="utf-8",index=False)
        return