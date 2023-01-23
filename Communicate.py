'''
THIS FILE IS SUPPOSE TO PROCESS THE TEXT DATA WITHIN THE SRS DOCUMENT.
STEP 1 IS TO EXTRACT ALL THE PAGES OF THE DOCUMENT INTO STRING AND THEN
PROCESS THAT STRING INTO A PROCESSED FORM THAT ONLY CONTAINS PERTINENT INFORMATION
'''
if __name__ == '__main__':
    import csv
    from collections import defaultdict
    import re
    import numpy as np
    import pandas as pd
    from pprint import pprint
    import PyPDF2
    import string
    import re
    import spacy
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag,ne_chunk

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
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from nltk.corpus import stopwords
    stopWords = stopwords.words('english')
    stopWords.extend(["from","subject","re","edu","use"])
    nlp = spacy.load("en_core_web_sm",disable=["parser,ner"])


    def pos_tagging(text):
        word_tokens = word_tokenize(text)
        return pos_tag(word_tokens)

    def named_entity_recognition(text):
        word_tokens = word_tokenize(text)
        word_pos = pos_tag(word_tokens)
        print(ne_chunk(word_pos))

    def chunking (text,grammar):
        word_tokens = word_tokenize(text)
        word_pos = pos_tag(word_tokens)
        chunnkParser = nltk.RegexpParser(grammar)
        tree = chunnkParser.parse(word_pos)
        for subtree in tree.subtrees():
            print(subtree)
        tree.draw()

    def removePunctuation(text):
        translator = str.maketrans('','',string.punctuation)
        return text.translate(translator)

    def removeWhiteSpaces(text):
        return " ".join(text.split())

    def sent_to_words(sentences):
        count = 0
        for sentence in sentences:
            count+=1
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def remove_stopwords(text):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in text]

    def lemmatization(texts, allowed_pos=["NOUN", "ADJ", "VERB", "ADV"]):
        count = 0
        text_out = list()
        for sent in texts:
            doc = nlp(" ".join(sent))
            text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_pos])
        return text_out


    #OPEN PDF
    pdfFileObj = open('UpdatedLogisticsSRS.pdf','rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    #String of our extracted text
    result =[]
    temp= ""

    #EXTRACT ALL PAGES
    for x in range(6,17):
        temp = pdfReader.getPage(x).extractText()
        # CONVERT TO LOWERCASE
        temp = temp.lower()
        # REMOVE NUMBERS
        temp = re.sub(r'\d+', '',temp)
        # REMOVE PUNCTUATION
        temp = removePunctuation(temp)
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

    #LEMMETIZATION
    lemma =  lemmatization(data_words_bigrams, allowed_pos=['NOUN', 'ADJ', 'VERB', 'ADV'])

    ''' 
    id2word serves as the dictionary that maps the words to their intenger ids within
    '''
    id2word = corpora.Dictionary(lemma)
    texts = lemma

    '''
    The corpus represents our documents from id2word in the format of a Bag of Words. 
    "A bag of words is a representation of text that describes the occurrence of words within a document." 
    source: https://www.mygreatlearning.com/blog/bag-of-words/#sh1
    '''
    corpus = [id2word.doc2bow(text) for text in texts]


    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=15,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    pprint(lda_model.print_topics())

    doc_lda = lda_model[corpus]
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    coherence_model_lda = CoherenceModel(model=lda_model, texts=lemma, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.display(vis_data)

    #Save data to disk we save the corpus,lda model, and the dictionary of words
    pyLDAvis.save_html(vis_data, "logistics_result")
    lda_model.save("logistics_srs_lda")
    corpora.MmCorpus.serialize("logistics_srs_corpus",corpus)
    id2word.save("logistics_srs_dictionary")
    print(id2word)
    print(corpus)
    print(id2word.token2id)
