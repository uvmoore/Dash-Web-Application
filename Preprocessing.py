"""FILE FOR TEXT PREPROCESSING SRS DOCUMENTS"""
import io
# PyMuPDF python Library
import pickle
import re
import fitz
import string
import spacy
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

stopWords = stopwords.words('english')
stopWords.extend(["from", "subject", "re", "edu", "use"])
nlp = spacy.load("en_core_web_sm", disable=["parser,ner"])


class Preprocess:
    def __init__(self):
        self.start_page = None
        self.end_page = None
        self.processed_text = []

    def get_start_page(self):
        """ Returns the start page number of document """
        return self.start_page

    def get_end_page(self):
        """ Returns the end page number of document """
        return self.end_page

    def get_dictionary(self):
        """Return the Dictionary for the Processed Document"""
        return self.id2word

    def get_corpus(self):
        """Return the Corpus for Processed Document"""
        return self.corpus

    def get_text(self):
        """Return the Text for Processed Document"""
        return self.lemma

    def set_start_page(self, start_page_num):
        """ Sets the Starting Page number of document """
        self.start_page = start_page_num

    def set_end_page(self, end_page_num):
        """ Sets the Ending Page number of the document """
        self.end_page = end_page_num

    def open_pdf(self, filepath):
        self.pdf = fitz.open(filepath)

    def open_pdf_stream(self, filestream):
        """ Opens PDF file from a stream """
        self.pdf = fitz.Document(stream=io.BytesIO(filestream))

    def process_pdf(self):
        """ Reads the PDF and extracts its contents """
        for x in range(self.start_page, self.end_page):
            current_page = self.pdf.load_page(x)
            # Read Current Page
            extracted_text = current_page.get_text("text")
            # Convert text to lower case
            extracted_text = extracted_text.lower()
            # Remove numbers from text
            extracted_text = re.sub(r'\d+', '', extracted_text)
            # Remove punctuation from text
            extracted_text = self._remove_punctuation(extracted_text)
            # Remove whitespaces
            extracted_text = self._remove_whitespaces(extracted_text)
            extracted_text = list(extracted_text.split(" "))
            extracted_text = list(filter(None, extracted_text))
            self.processed_text.append(extracted_text)

    def form_n_grams(self):
        data_words = list(self._sent_to_words(self.processed_text))
        data_words = self._remove_stopwords(data_words)
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[data_words], min_count=5, threshold=100)
        bigram_model = gensim.models.phrases.Phraser(bigram)
        trigram_model = gensim.models.phrases.Phraser(trigram)
        self.data_words_bigrams = [bigram_model[doc] for doc in data_words]
        self.data_words_trigrams = [x for x in self.data_words_bigrams if x != []]

    def form_topic_model_inputs(self):
        self.lemma = self._lemmatization(self.data_words_bigrams)
        self.id2word = corpora.Dictionary(self.lemma)
        self.corpus = [self.id2word.doc2bow(word) for word in self.lemma]

    def save_corpus(self, filepath):
        """Save Corpus for Processed Document"""
        corpora.MmCorpus.serialize(filepath, self.corpus)

    def save_dictionary(self, filepath):
        """Save Dictionary for Processed Document"""
        self.id2word.save(filepath)

    def save_text(self, filepath):
        file = open(filepath, 'wb')
        pickle.dump(self.lemma, file)

    def _remove_punctuation(self, supplied_text):
        """Removes punctuation"""
        cleaned_text = str.maketrans('', '', string.punctuation)
        return supplied_text.translate(cleaned_text)

    def _remove_whitespaces(self, supplied_text):
        """Removes redundant white spaces"""
        return " ".join(supplied_text.split())

    def _sent_to_words(self, sentences):
        count = 0
        for sentence in sentences:
            count += 1
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    def _remove_stopwords(self, supplied_text):
        """remove stop words from text"""
        return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in supplied_text]

    def _lemmatization(self, supplied_text):
        allowed_pos = ["NOUN", "ADJ", "VERB", "ADV"]
        text_out = list()
        for sent in supplied_text:
            doc = nlp(" ".join(sent))
            text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_pos])
        return text_out