import random
from collections import Counter

import numpy as np
import numpy.typing as npt

# Hyperparameters
eta = 0.01
order_of_ngrams = 3
epochs = 4


### Metrics for Binary Classification

def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)

def f_measure(beta: float, tp: int, fp: int, fn: int) -> float:
    return (1 + beta**2) * (precision(tp, fp) * recall(tp, fn)) / (beta**2 * precision(tp, fp) * recall(tp, fn))

def f1(tp: int, fp: int, fn: int) -> float:
    return f_measure(1, tp, fp, fn)


### Micro-Averaged Metrics

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


### Macro-Averaged Metrics

def macro_precision(tp: "dict[str, int]", fp: "dict[str, int]") -> float:
    n = len(tp)
    return (1 / n) * sum([precision(tp[c], fp[c]) for c in tp.keys()])

def macro_recall(tp: "dict[str, int]", fn: "dict[str, int]") -> float:
    n = len(tp)
    return (1 / n) * sum([recall(tp[c], fn[c]) for c in tp.keys()])

def  macro_f1(tp: "dict[str, int]", fp: "dict[str, int]", fn: "dict[str, int]") -> float:
    n = len(tp)
    return 2 * (macro_precision(tp, fp) * macro_recall(tp, fn)) / (macro_precision(tp, fp) + macro_recall(tp, fn))


# Preprocessing

def extract_ngrams(x: str, n=3) -> "list[str]":
    return [''.join(s) for s in (zip(*[x[i:] for i in range(n)]))]

def to_onehot_vector(lang: str, langs: list[str]) -> np.ndarray:
    y = np.zeros(len(langs))
    y[langs.index(lang)] = 1
    return y

def vectorize_ngrams(counter: dict[str, int], feature_map: dict[str, int]) -> np.ndarray:
    feature_vector = np.zeros(len(feature_map))
    for ngram, count in counter.items():
        if ngram in feature_map:
            feature_vector[feature_map[ngram]] = count
    return feature_vector

def preprocess_training_observations(fn: str, n: int=1) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, int]]:
    langs = set()
    features = set()
    obs = []
    with open(fn) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for lang, doc in reader:
            langs.add(lang)
            ngrams = extract_ngrams(doc, n)
            features = features | set(ngrams)
            obs.append((lang, ngrams))
    feature_map = {feature: idx for idx, feature in enumerate(features)}
    lang_list = list(langs)
    lang_map = {lang: to_onehot_vector(lang, lang_list) for lang in langs}
    obs = [(lang_map[lang], vectorize_ngrams(Counter(ngrams), feature_map)) for (lang, ngrams) in obs]
    print(f"{len(obs)} training observations.")
    return obs, lang_map, feature_map

def preprocess_test_observations(fn: str, feature_map: dict[str, int], lang_map: dict[str, np.ndarray], n: int=1) -> list[tuple[np.ndarray, np.ndarray]]:
    with open(fn) as fin:
        obs = []
        reader = csv.reader(fin, delimiter='\t')
        for lang, doc in reader:
            ngrams = extract_ngrams(doc, n)
            x = vectorize_ngrams(Counter(ngrams), feature_map)
            try:
                obs.append((lang_map[lang], x))
            except KeyError:
                print(f"Unkown langugae {lang}!")
    print(f"{len(obs)} test observations.")
    return obs


### Classification Function

def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    return(np.exp(z - np.max(z)) / np.exp(z - np.max(z)).sum())

def grad(W: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y_hat = softmax(np.dot(W,x))
    return np.outer(np.subtract(y_hat,y),x)

def train(observations: tuple[np.ndarray, np.ndarray], epochs: int = 1) -> np.ndarray:
    N = observations[0][1].shape[0] + 1
    K = observations[0][0].shape[0]
    W = np.zeros((K,N))

    for epoch in range(epochs):
        random.shuffle(observations)
        tp, fp, fn = Counter(), Counter(), Counter()
        for y, x in observations:
            W = W - grad(W, y, np.insert(x,0,1))

            ref_lang = np.argmax(y)
            y_hat = softmax(np.dot(W,np.insert(x,0,1)))
            hyp_lang = np.argmax(y_hat)

            if hyp_lang == ref_lang:
                tp[ref_lang] += 1
            else:
                fp[hyp_lang] += 1
                fn[ref_lang] += 1
    return W

def classify(W: np.ndarray, x: np.ndarray) -> np.intp:
    return np.argmax(softmax(np.dot(W,x)))


### Evaluate the Classifier

def evaluate(train_set, test_set, epochs=3):
    print("Training model")
    W = train(train_set, epochs=epochs)
    print("\nCLASSIFY")
    tp, fp, fn = Counter(), Counter(), Counter()
    for ref_lang_vec, x in test_set:
        ref_lang = np.argmax(ref_lang_vec)
        x = np.insert(x, 0, 1)
        hyp_lang = classify(W, x)
        if hyp_lang == ref_lang:
            tp[ref_lang] += 1
        else:
            fp[hyp_lang] += 1
            fn[ref_lang] += 1

    print(f'macro-averaged F1:\t\t{macro_f1(tp, fp, fn):.3f}')
    print(f'micro-averaged F1:\t\t{micro_f1(tp, fp, fn):.3f}')

train_set, lang_map, feature_map = preprocess_training_observations("train.tsv", n=order_of_ngrams)
test_set = preprocess_test_observations("dev.tsv", feature_map, lang_map, n=order_of_ngrams)
evaluate(train_set, test_set, epochs=epochs)
