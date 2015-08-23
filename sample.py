__author__ = 'dowling'
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(name)-18s: %(message)s', level=logging.DEBUG)

ln = logging.getLogger(__name__)

from classify import train_classifier, evaluate_classifier
from data.rcv1corpus import rcv1_train, rcv1_train_target, rcv1_test, rcv1_test_target

from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel

from gensim.corpora.dictionary import Dictionary

def main():
    try:
        dictionary = Dictionary.load_from_text("dictionary.txt")
    except:
        dictionary = Dictionary(rcv1_train)
        dictionary.filter_extremes()
        dictionary.save_as_text("dictionary.txt")

    class RCV1BowCorpus(object):
        def __iter__(self):
            for document in rcv1_train:
                yield dictionary.doc2bow(document)

    ln.debug("Training model on %s documents" % len(rcv1_train))
    try:
        vector_model = LsiModel.load("lsi_model")
    except:
        vector_model = LsiModel(corpus=RCV1BowCorpus(), num_topics=100, id2word=dictionary)
        vector_model.save("lsi_model")

    def get_lsi_features(text):
        """
        Must return either numpy array or dictionary
        """
        res = vector_model[dictionary.doc2bow(text)]
        return dict(res)

    def get_bow_features(text):
        return dict(dictionary.doc2bow(text))

    clf = train_classifier(train_samples=rcv1_train, train_targets=rcv1_train_target, get_features=get_lsi_features,
                           classifier="sgd")

    evaluate_classifier(clf, rcv1_test, rcv1_test_target, get_features=get_lsi_features)

if __name__ == "__main__":
    main()