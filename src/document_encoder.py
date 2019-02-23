"""
Document encoder
Transforms documents into numeric features.
"""

import numpy as np
import string

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import sklearn.feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class DocumentEncoder():

    punctuation = set(string.punctuation)

    def __init__(self, texts=None):
        self.texts = list(texts) if texts is not None else None ## Full dataset of texts for constructing vocabulary
        self.load_resources()

        self.counter = DocumentEncoder.fit_counter(self.texts) if texts is not None else None

    def load_resources(self):
        nltk.download('punkt')

    def featurize(self, X):
        data = None
        for idx, sample in enumerate(X):
            featurized = self.featurize_sample(sample)
            if data is None:
                data = np.ndarray((len(X), *featurized.shape), dtype=np.float32)
            data[idx] = featurized
            if idx > 0 and idx % 1000 == 0:
                print('Progress: {} / {}'.format(idx+1, len(X)))
        
        return data

    def featurize_sample(self, x):
        featurized = self.featurize_sample_nltk(x)
        if self.counter is not None:
            featurized = np.concatenate((featurized, self.sample_count(x)))
        return featurized

    def __call__(self, x):
        return self.featurize_sample(x)

    def sample_count(self, x):
        return self.counter.transform([x.get_text_cleaned()]).toarray().flatten()

    def featurize_sample_nltk(self, x):
        ## Average sentence length
        sentences = sent_tokenize(x.get_text())
        num_sentences = len(sentences)
        
        sent_length_words = [len(word_tokenize(s)) for s in sentences]
        sent_length_words = (sum(sent_length_words) / len(sent_length_words)) if len(sent_length_words) > 0 else 0

        sent_length_chars = [len(s) for s in sentences]
        var_sent_length_chars = np.var(sent_length_chars)
        sent_length_chars = (sum(sent_length_chars) / len(sent_length_chars)) if len(sent_length_chars) > 0 else 0

        ## Average word length
        words = [w for w in word_tokenize(x.get_text_cleaned())]
        word_length = [len(w) for w in words]

        word_length_var = np.var(word_length)     ## Variance of word length
        word_length = (sum(word_length) / len(word_length)) if len(word_length) > 0 else 0  ## Average word length

        ## Frequency of punctuation
        punct_freq = [1 if c in DocumentEncoder.punctuation else 0 for c in x.get_text() if c != ' ']
        punct_freq = sum(punct_freq) / len(punct_freq) if len(punct_freq) > 0 else 0

        ## Frequency of capital case letters
        capital_freq = [1 if c.isupper() else 0 for c in x.get_text() if c.isalpha()]
        capital_freq = sum(capital_freq) / len(capital_freq) if len(capital_freq) > 0 else 0

        ## Ratio of types / atoms (vocabulary richness)
        previously_seen = set()
        total_types, total_atoms = 0, 0
        for w in words:
            if w not in previously_seen:
                previously_seen.add(w)
                total_types += 1
            total_atoms += 1

        return np.array([
                num_sentences, sent_length_words, sent_length_chars, var_sent_length_chars,
                word_length, word_length_var, punct_freq, capital_freq,
                total_types / total_atoms if total_atoms > 0 else 0],
                \
                dtype=np.float32
            )

    @staticmethod
    def top_tfidf_terms(tfidf, text, n=10):
        matrix = tfidf.transform([text])
        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(matrix.toarray()).flatten()[::-1]
        ## NOTE toarray() transforms sparse array to dense array, which kills RAM...

        return feature_array[tfidf_sorting[:n]]

    @staticmethod
    def fit_tfidf(texts):
        print('Fittinf TF-IDF for {} texts...'.format(len(texts)))
        tfidf = TfidfVectorizer(tokenizer=DocumentEncoder.tokenize, stop_words='english', ngram_range=(1,2))
        ## NOTE can also easily use character ngrams with analyzer='char'
        return tfidf.fit(texts) # returns self

    @staticmethod
    def fit_counter(texts, **kwargs):
        print('Fitting CountVectorizer for {} texts'.format(len(texts)))
        counter = CountVectorizer(
            tokenizer=DocumentEncoder.tokenize,
            stop_words=[DocumentEncoder.tokenize(w)[0] for w in sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS],
            ngram_range=(1,2), max_features=50, max_df=0.95,
            **kwargs
        )

        return counter.fit(texts)

    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        return [stemmer.stem(tok) for tok in tokens]

    @staticmethod
    def strip_punctuation(text):
        trans_table = dict.fromkeys(map(string.punctuation), None)
        ## or: trans_table = str.maketrans('','',string.punctuation)
        return text.translate(trans_table)
