### Preprocessing

from collections.abc import Generator
import re

def preprocess(doc: str) -> Generator[str, None, None]:
    stop_words = ["I","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]
    suffixes = ["s","ed","ing", "tion","tions","or","ors","able","tive","tively","ness"]

    tokenized_doc = re.findall(r'\w+',doc)

    for token in tokenized_doc:
      token_is_stop_word = False
      for stop_word in stop_words:
        if token == stop_word:
          token_is_stop_word = True
      if token_is_stop_word == False:
        for suffix in suffixes:
          if token.endswith(suffix):
            suffix_length = len(suffix)
            token = token[:-suffix_length]
        yield token

doc = None
with open('test/1.txt', 'r', encoding ='latin-1') as f:
    doc = f.read()
for token in preprocess(doc):
    print(token, end = ' ')

from google.colab import drive
drive.mount('/content/drive')

### Training the Naive Bayes Classifier

from collections import defaultdict, Counter
from math import log
from math import abs
import pickle
from operator import itemgetter
from tqdm import tqdm

def extract_ngrams(x: 'list[str]', n=1) -> "list[str]":
    tokenized_doc = preprocess(x)
    tokenized_doc.split(" ")
    ngrams = []
    for i in range(len(tokenized_doc)-(n-1)):
      ngram = tokenized_doc[i,i+n]
      ngrams.append(ngram)
    return ngrams

def smoothed_log_likelihood(w: str, c: str, k: int, count: 'DefaultDict[str, Counter]', vocab: "set[str]") -> float:
    sum_wv = 0
    for vw in vocab:
      sum_wv += count[vw][c]+k
    log_likelihood = log((count[w][c]+k)/sum_wv)
    return log_likelihood

def train_nb(docs: "list[tuple[list[str], str]]", k: int = 1, n: int = 1, model_save_path = "model.pkl") -> "tuple[dict[str, float], DefaultDict[str, DefaultDict[str, float]], set[str], set[str]]":

    vocab = set()
    classes = set()
    docs_by_class = defaultdict(list)
    for doc, c in docs:
        # Represent documents as collections of ngrams (default n=3).
        ngrams = extract_ngrams(doc, n)
        vocab |= set(ngrams)
        docs_by_class[c].append(ngrams)
        classes.add(c)
    num_docs = len(docs)
    num_docs_in_class = {}
    log_prior = {}
    count = defaultdict(Counter)
    log_likelihood = defaultdict(defaultdict)
    for c, documents in tqdm(docs_by_class.items(), position = 0):
        for d in documents:
            for w in d:
                count[w][c] += 1
        num_docs_in_class[c] = len(documents)
        for w in tqdm(vocab, position = 0):
            log_likelihood[w][c] = smoothed_log_likelihood(w, c, k, count, vocab)

    with open(model_save_path, 'wb') as f:
        pickle.dump((log_prior, log_likelihood, classes, vocab, count), f)
        print('Model Saved!')
    return (log_prior, log_likelihood, classes, vocab, count)

def prepare_data(index_file, data_dir):
    l = None
    with open(index_file, 'r') as f:
        l = f.readlines()
        l = [(data_dir + i.split()[0], i.split()[1]) for i in l]

    data = []
    doc = None

    for file, label in l:
        with open(file, 'r', encoding ='latin-1') as f:
            doc = f.read()
            data.append(([token for token in preprocess(doc)], label))
    return data

train_data = prepare_data('index-train.txt', 'train/')
print(len(train_data))
x = train_nb(train_data)


### Using the Naive Bayes Classifier to Make Predictions

def classify(testdoc: str, log_prior: "dict[str, float]", log_likelihood: "DefaultDict[str, DefaultDict[str, float]]", classes: "set[str]", vocab: "set[str]", count: "DefaultDict[str, Counter]", k: int=1, n: int=1) -> str:
    doc = extract_ngrams(testdoc, n)
    class_sum = {}
    for c in classes:
        class_sum[c] = log_prior[c]

test_data = prepare_data('index-test.txt', 'test/')
with open('model.pkl', 'rb') as f:
    log_prior, log_likelihood, classes, vocab, count = pickle.load(f)
print("Testing Trained Naive Bayes Classifier\n\n")

for i in range(3):
    print(f'Prediction {i + 1}: {classify(test_data[i][0], log_prior, log_likelihood, classes, vocab, count)}')
    print(f'Actual {i + 1}: {test_data[i][1]}\n')


### Evaluating Naive Bayes Classifier

def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)

def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)

def f_measure(beta: float, tp: int, fp: int, fn: int) -> float:
    return (1 + beta**2) * (precision(tp, fp) * recall(tp, fn)) / (beta**2 * precision(tp, fp) * recall(tp, fn))

def f1(tp: int, fp: int, fn: int) -> float:
    return f_measure(1, tp, fp, fn)

def micro_precision(tp: "dict[str, int]", fp: "dict[str, int]") -> float:
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    return tp_sum / (tp_sum + fp_sum)

def micro_recall(tp: "dict[str, int]", fn: "dict[str, int]") -> float:
    tp_sum = sum(tp.values())
    fn_sum = sum(fn.values())
    return tp_sum / (tp_sum + fn_sum)

def micro_f1(tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
    mp = micro_precision(tp, fp)
    mr = micro_recall(tp, fn)
    return 2 * (mp * mr) / (mp + mr)

def macro_precision(tp: "dict[str, int]", fp: "dict[str, int]") -> float:
    n = len(tp)
    return (1 / n) * sum([precision(tp[c], fp[c]) for c in tp.keys()])

def macro_recall(tp: "dict[str, int]", fn: "dict[str, int]") -> float:
    n = len(tp)
    return (1 / n) * sum([recall(tp[c], fn[c]) for c in tp.keys()])

def  macro_f1(tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
    n = len(tp)
    return 2 * (macro_precision(tp, fp) * macro_recall(tp, fn)) / (macro_precision(tp, fp) + macro_recall(tp, fn))

tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

for testdoc, truth in test_data:
    predicted = classify(testdoc, log_prior, log_likelihood, classes, vocab, count)
    if predicted == truth:
        tp[truth] += 1
    else:
        fp[predicted] += 1
        fn[truth] += 1
print(f'Macro-averaged precision:\t{macro_precision(tp, fp)}')
print(f'Macro-averaged recall:\t{macro_recall(tp, fn)}')
print(f'Macro-averaged F1:\t{macro_f1(tp, fp, fn)}')
print(f'Micro-averaged precision:\t{micro_precision(tp, fp)}')
print(f'Micro-averaged recall:\t{micro_recall(tp, fn)}')
print(f'Micro-averaged F1:\t{micro_f1(tp, fp, fn)}')

