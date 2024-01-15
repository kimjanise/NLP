from collections.abc import Generator
import re
from collections import defaultdict, Counter
from math import log
import pickle
from operator import itemgetter

class Solution:
    def preprocess(self, doc: str) -> Generator[str, None, None]:
        """
        Function to preprocess and tokenize the document
        :param doc: The document being preprocessed
        :type doc: str
        :return: Generator for every new token of the preprocessed text
        :rtype: Generator[str, None, None]
        
        """
        # TODO: fill in your preprocessing steps
        stop_words = ["I","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]
        suffixes = ["s","ed","ing","tion","tions","or","ors","able","tive","tively","ness"]
    	# separation of punctuation from words
        tokenized_doc = re.findall(r'\w+', doc)
    	#removal of stop words
        for token in tokenized_doc:
            token_is_stop_word = False
            for stop_word in stop_words:
                if token == stop_word:
                    token_is_stop_word = True
            if token_is_stop_word == False:
        		#stemming/lemmatization
                for suffix in suffixes:
                    if token.endswith(suffix):
                        suffix_length = len(suffix)
                        token = token[:-suffix_length]
                if(len(token) != 0):
                    yield token

    def extract_ngrams(self, x: str, n=1) -> "list[str]":
        """
        Extract n-grams from the input string
        
        :param x: The string from which the n-grams must be extracted
        :type x: str
        :param n: The order of the n-gram that must be extracted
        :type n: int
        :return: The list of n-grams
        :rtype: list[str]
        """
        # TODO: fill in function to return a list of all n-grams in input text
        return x


    def smoothed_log_likelihood(self, w: str, c: str, k: int, count: 'DefaultDict[str, Counter]', vocab: "set[str]") -> float:
        # TODO: fill in function to return the log likelihood with add-k smoothing 
        """
        Calculate the smoothed add-k log likelihood value
        
        :param w: The word for which the value must be computed
        :type w: str
        :param c: The class of the current document
        :type c: str
        :param k: The k value to be used in the add-k smoothing
        :type k: str
        :param count: The order of the gram that must be extracted
        :type count: int
        :param vocab: The order of the gram that must be extracted
        :type vocab: int
        :return: The log likelihood value
        :rtype: float
        """
        sum_wv = 0
        for vw in vocab:
            sum_wv += count[vw][c]+k
        log_likelihood = log((count[w][c]+k)/sum_wv)
        return log_likelihood

    def train_nb(self, docs: "list[tuple[list[str], str]]", k: int = 1, n: int = 1, model_save_path = "model.pkl") -> "tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]":
        """
        Train a Naive-Bayes model

        :param docs: The documents (tokenized into a list of tokens), each associated with a class label (document, label)
        :type docs: list[tuple[list[str], str]]
        :param k: The value added to the numerator and denominator to smooth likelihoods
        :type k: int
        :para n: the order of ngrams
        :type b: int
        :return: The log priors, log likelihoods, the classes, the vocabulary, and the counts for the model at a tuple
        :rtype: tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]], DefaultDict[str, Counter]
        """
        # Inialized vocab (the vocabulary) and docs_by_class (a dictionary in which
        # class labels are keys and lists of documents are values).
        vocab = set()
        classes = set()
        docs_by_class = defaultdict(list)
        # Populate vocab and docs_by_class.
        for doc, c in docs:
            # Represent documents as collections of ngrams (default n=3).
            ngrams = self.extract_ngrams(doc, n)
            vocab |= set(ngrams)
            docs_by_class[c].append(ngrams)
            classes.add(c)
        # Total number of documents in the collection
        num_docs = len(docs)
        # Number of documents in each class (with each label)
        num_docs_in_class = {}
        # The log priors for each class
        log_prior = {}
        # counts of times a label c occurs with an ngram w
        count = defaultdict(Counter)
        log_likelihood = defaultdict(defaultdict)
        # For each 
        for c, documents in docs_by_class.items():
            # Iterate over the documents of class c, accumulating <word, class> counts in counts
            for d in documents:
                for w in d:
                    count[w][c] += 1
            # Calculate the number of documents with class label c
            num_docs_in_class[c] = len(documents)
            
            # TODO: calculate the log prior for the class
            log_prior[c] = log(abs(num_docs_in_class[c])/abs(num_docs))

            # Calculate the log likelihood for each (w|c). Smooth by k.
            for w in vocab:
                log_likelihood[w][c] = self.smoothed_log_likelihood(w, c, k, count, vocab)
                
                
        with open(model_save_path, 'wb') as f:
            pickle.dump((log_prior, log_likelihood, classes, vocab, count), f)
            print('Model Saved!')
        return (log_prior, log_likelihood, classes, vocab, count)


    def classify(self, testdoc: str, log_prior: "dict[str, float]", log_likelihood: "DefaultDict[str, DefaultDict[str, float]]", classes: "set[str]", vocab: "set[str]", count: "DefaultDict[str, Counter]", k: int=1, n: int=1) -> str:
        """Given a trained NB model (log_prior, log_likelihood, classes, and vocab), returns the most likely label for the input document.

        :param textdoc str: The test document.
        :param log_prior dict[str, float]: The log priors of each category. Categories are keys and log priors are values.
        :param log_likelihood DefaultDict[str, DefaultDict[str, float]]: The log likelihoods for each combination of word/ngram and class.
        :param classes set[str]: The set of class labels (as strings).
        :param vocab set[str]: The set of words/negrams in the vocabulary.
        :param count DefaultDict[str, Counter]: The counts of each token per class calculated during training
        :param k int: the value added in smoothing.
        "param n int: the order of ngrams.
        :return: The best label for `testdoc` in light of the model.
        :rtype: str
        """
        # Extract a set of ngrams from `testdoc`
        doc = self.extract_ngrams(testdoc, n)
        # Initialize the sums for each class. These will be the "scores" based on which class will be assigned.
        class_sum = {}

        
        # Iterate over the classes, computing `class_sum` for each
        
        for c in classes:
            # Initialize `class_sum` with the log prior for the class
            class_sum[c] = log_prior[c]
            # TODO: Add the likelihood for each in-vocabulary word/ngram in the document to the class sum
            for w in doc:
                if w in vocab:
                    class_sum[c] += log_likelihood[w][c]
                else:
                    class_sum[c] += self.smoothed_log_likelihood(w,c,k,count,vocab)

        # TODO: Return the best label with the highest probability
        hc = "politics"
        hp = class_sum[hc]
        for c in classes:
            if class_sum[c] > hp:
                hp = class_sum[c]
                hc = c
        return hc
