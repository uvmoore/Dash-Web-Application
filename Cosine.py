# Gensim
import copy
import gensim
import pandas as pd
import gensim.corpora as corpora
from SRSLDA import lda_model, corpora

'''Calulate the cosine similarity for each topic and attack pattern and save the results to a list'''
def caluclate_cosine_similarity():
    result = {}
    for x in range(0, 10):
        for y in range(0, 546):
            result.update({y: gensim.matutils.cossim(srs_vector[x], capec_vector_list[y])})
        result_list.append(copy.deepcopy(result))
        result.clear()

def create_CAPEC_vectors():
    capec_vector = []
    capec_num = 1
    vector_list = []
    for x in range(1, 547):
        capec_lda = gensim.models.LdaModel.load("CAPEC/LDA/capec_lda_" + str(capec_num))
        corpus2 = corpora.MmCorpus("CAPEC/CORPUSES/capec_corpus_" + str(capec_num))
        for topic_num in range(0, 4):
            capec_vector.extend(capec_lda.get_topic_terms(topicid=topic_num, topn=10))
        vector_list.append(capec_vector[:])
        capec_vector.clear()
        capec_num += 1
    return vector_list

def get_CAPEC_IDs(capec_list):
    CAPEC_IDs = pd.read_csv("CAPEC/Comprehensive CAPEC Dictionary.csv",usecols=["ID"])
    list_with_topics_and_capec_ids = []
    for topic in capec_list:
        topic_list = []
        for value in topic:
            index = value[0]-1
            new_tuple = tuple((CAPEC_IDs.iloc[index].values[0],value[1]))
            topic_list.append(new_tuple)
        list_with_topics_and_capec_ids.append(copy.deepcopy(topic_list))
    return list_with_topics_and_capec_ids

'''Global Variables'''

capec_vector_list = []
srs_vector = []
similarity_results = {}
csv_filename = "placeholder.csv"
srs_lda_filename = lda_model
srs_corpus_filename = corpora
capec_num = 1

srs_lda = gensim.models.LdaModel.load(srs_lda_filename)
srs_corpus = corpora.MmCorpus(srs_corpus_filename)

#Load in the LDA model from the SRS document into
for topic_num in range(0,10):
    srs_vector.append(srs_lda.get_topic_terms(topicid=topic_num,topn=30))

# Load in the capec lda models into vectors
capec_vector_list = create_CAPEC_vectors()


result_list = []

caluclate_cosine_similarity()

# Sort the values of the list from highest similarity value to lowest similarity value.
sorted_lists = []
for y in range(0,10):
     sorted_lists.append(sorted(result_list[y].items(),key=lambda x:x[1],reverse=True))

# Returns a new list with the CAPEC IDS of the Attack Pattern for all topics

result = get_CAPEC_IDs(sorted_lists)


#Convert the list of lists into a Pandas Dataframe. For easier manipulation.
pandas_frame = pd.DataFrame(result)

#Transpose the Data Frame
pandas_frame = pandas_frame.transpose()

#Rename the columns of the Pandas DataFrame
for x in range(0,10):
    pandas_frame.rename(columns={x:"Topic-"+str(x+1)},inplace=True)



#Save the results to a csv file.
pandas_frame.to_csv("resources/"+csv_filename,encoding="utf-8",index=False)

#print the results to the console

for x in range (0,10):
    print("Topic Number "+str(x))
    for y in range (0,10):
        print(sorted_lists[x][y])

