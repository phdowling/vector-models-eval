__author__ = 'dowling'
import logging
ln = logging.getLogger(__name__)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

import numpy as np

classifiers = {
    "sgd": lambda: SGDClassifier(loss="log"),
    "nb": lambda: MultinomialNB(),
    "svc": lambda: LinearSVC()
}


class ArrayVectorizer(object):
    """
    Just a wrapper to ensure arrays have the right format
    """
    @staticmethod
    def fit_transform(X):
        return np.vstack(
            map(lambda x: np.array(x).reshape(1, len(x)), X)
        )

    @staticmethod
    def transform(X):
        return ArrayVectorizer.fit_transform(X)

def select_vectorizer(feature_type):
    if feature_type == dict:
        return DictVectorizer()
    elif feature_type == str:
        return CountVectorizer()
    elif feature_type == list:
        return ArrayVectorizer()
    elif feature_type == np.ndarray:
        return ArrayVectorizer()
    else:
        raise ValueError("Can't vectorize features!")


feature_vectorizer = None
def train_classifier(train_samples, train_targets, get_features=None, classifier="sgd"):
    global feature_vectorizer
    ln.info("Training %s classifier on %s train samples." % (classifier, len(train_samples)))

    if get_features is None:  # use bow
        get_features = lambda sample: sample

    # figure out the feature type and select vectorization scheme
    first = get_features(train_samples[0])
    feature_vectorizer = select_vectorizer(type(first))
    ln.debug("Selected %s for vectorization." % feature_vectorizer)

    samples_feats = feature_vectorizer.fit_transform(map(get_features, train_samples))

    ln.debug("Beginning training.")
    clf = OneVsRestClassifier(classifiers[classifier](), n_jobs=1)
    clf.fit(samples_feats, train_targets)
    ln.info("Training completed.")
    return clf


def evaluate_classifier(clf, test_samples, test_targets, get_features=None):
    global feature_vectorizer
    ln.info("Beginning evaluation on %s samples." % len(test_samples))
    if get_features is None:  # use bow
        get_features = lambda sample: sample

    samples_feats = feature_vectorizer.transform(map(get_features, test_samples))
    results = clf.predict(samples_feats)
    all_labels = clf.classes_

    g_tp = 0
    g_fp = 0
    g_fn = 0
    f1_scores_micro = []
    for idx, target_labels in enumerate(test_targets):
        predicted_labels = results[idx]
        tp = 0
        fp = 0
        fn = 0
        for label in all_labels:
            if label in target_labels and label in predicted_labels:
                tp += 1
            elif label in target_labels and label not in predicted_labels:
                fn += 1
            elif label not in target_labels and label in predicted_labels:
                fp += 1

        try:
            precision = float(tp) / float(tp + fp)
            recall = float(tp) / float(tp + fn)
        except ZeroDivisionError:
            continue
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            ln.warn("No true positives at all!")
            continue
        f1_scores_micro.append(f1)
        g_tp += tp
        g_fp += fp
        g_fn += fn

    g_precision = float(g_tp) / float(g_tp + g_fp)
    g_recall = float(g_tp) / float(g_tp + g_fn)

    F1 = 2 * (g_precision * g_recall) / (g_precision + g_recall)
    average_f1 = sum(f1_scores_micro) / float(len(f1_scores_micro))

    print "Global F1 score: %s\nFine-grained average F1 score: %s" % (F1, average_f1)
