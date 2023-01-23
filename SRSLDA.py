'''
THIS FILE IS SUPPOSE TO PROCESS THE TEXT DATA WITHIN THE SRS DOCUMENT.
STEP 1 IS TO EXTRACT ALL THE PAGES OF THE DOCUMENT INTO STRING AND THEN
PROCESS THAT STRING INTO A PROCESSED FORM THAT ONLY CONTAINS PERTINENT INFORMATION
'''
if __name__ == '__main__':
    import Cosine
    import csv
    from collections import defaultdict
    import re
    import numpy as np
    import pandas as pd
    import fitz
    import string
    import re
    import spacy
    import nltk
    import Cosine
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag, ne_chunk
    from main import newFile, pageCount

    # Plotting tools
    import pyLDAvis
    import pyLDAvis.gensim_models  # don't skip this
    import matplotlib.pyplot as plt

    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nltk.corpus import stopwords

    stopWords = stopwords.words('english')
    stopWords.extend(["from", "subject", "re", "edu", "use"])
    nlp = spacy.load("en_core_web_sm", disable=["parser,ner"])


    def pos_tagging(text):
        word_tokens = word_tokenize(text)
        return pos_tag(word_tokens)


    def named_entity_recognition(text):
        word_tokens = word_tokenize(text)
        word_pos = pos_tag(word_tokens)
        print(ne_chunk(word_pos))


    def chunking(text, grammar):
        word_tokens = word_tokenize(text)
        word_pos = pos_tag(word_tokens)
        chunkParser = nltk.RegexpParser(grammar)
        tree = chunkParser.parse(word_pos)
        for subtree in tree.subtrees():
            print(subtree)
        tree.draw()


    def removepunctuation(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)


    def removeWhiteSpaces(text):
        return " ".join(text.split())


    def sent_to_words(sentences):
        count = 0
        for sentence in sentences:
            count += 1
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


    def remove_stopwords(text):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in text]


    def lemmatization(texts, allowed_pos=["NOUN", "ADJ", "VERB", "ADV"]):
        count = 0
        text_out = list()
        for sent in texts:
            doc = nlp(" ".join(sent))
            text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_pos])
        return text_out


    def compute_coherence_values(dictionary, corpus, limit, start=2, step=1):
        coherence_values = []
        lda_model_list = []
        for num_topics in range(start, limit, step):
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                                        random_state=100, update_every=1,
                                                        chunksize=100, passes=10, alpha='auto', per_word_topics=True)
            lda_model_list.append(lda_model)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=lemma, dictionary=id2word, coherence='c_v')
            coherence_values.append(coherence_model_lda.get_coherence())
        return lda_model_list, coherence_values






    # String of our extracted text
    result = []
    temp = ""

    # Variables to hold LDA models and corresponding coherence values
    model_list = []
    coherence_values = []
    filename = "placeholder"
    pdf = newFile

    max = pageCount
    min = pageCount - 11

    # I chose that min count due to the range provided oringianlly being 35 - 46.
    # I am assuming a range of 11 pages should be good
    # OPEN PDF


    # EXTRACT PAGES
    for x in range(min, max):
        page = pdf.load_page(x)
        temp = page.get_text("text")
        # CONVERT TO LOWERCASE
        temp = temp.lower()
        # REMOVE NUMBERS
        temp = re.sub(r'\d+', '', temp)
        # REMOVE PUNCTUATION
        temp = removepunctuation(temp)
        # REMOVE WHITESPACES
        temp = removeWhiteSpaces(temp)
        temp = list(temp.split(" "))
        temp = list(filter(None, temp))
        result.append(temp)

    data_words = list(sent_to_words(result))
    # REMOVE DEFAULT STOPWORDS
    data_words_nostops = remove_stopwords(data_words)

    bigram = gensim.models.Phrases(data_words_nostops, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words_nostops], min_count=5, threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    data_words_bigrams = [x for x in data_words_bigrams if x != []]

    # LEMMETIZATION
    lemma = lemmatization(data_words_bigrams, allowed_pos=['NOUN', 'ADJ', 'VERB', 'ADV'])

    ''' 
    id2word serves as the dictionary that maps the words to their integer ids within
    '''
    id2word = corpora.Dictionary(lemma)
    texts = lemma

    corpus = [id2word.doc2bow(text) for text in texts]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, start=2, limit=42,
                                                            step=1)
    limit = 42
    start = 2
    step = 1
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence values", loc='best')
    plt.title("Coherence Scores")
    plt.show()
    plt.savefig("SRS/Figures/"+filename+".png")


    for num_topics, cv in zip(x, coherence_values):
        print("Num Topics =", num_topics, " has Coherence Value of", round(cv, 4))

    max_value = 0
    for y in range(0, len(coherence_values)-1):
        if max_value < coherence_values[y]:
            max_value = coherence_values[y]

    print("Max Coherence value is "+str(max_value)+" at index: " + str(coherence_values.index(max_value)))
    lda_model = model_list[coherence_values.index(max_value)]
    lda_model.save("resources/"+filename)
    corpora.MmCorpus.serialize("resources/"+filename, corpus)
    id2word.save("SRS/SRS LDA Topic Model/DICTIONARIES/"+filename+"_dictionary")
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word,mds='mmds')
    pyLDAvis.save_html(vis_data,"SRS/PYLDA/"+filename+"_visualization")

    script_f = 'Cosine.py'
    exec(open(script_f).read())
