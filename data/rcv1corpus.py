__author__ = 'dowling'
import logging
ln = logging.getLogger(__name__)
from collections import defaultdict
from random import shuffle, seed

_DEBUG_MODE = False

class RCV1Corpus(object):
    def __init__(self, path):
        self.path = path
        self.document_topics = defaultdict(list)
        ln.debug("Reading topics from file.")
        with open(self.path + "/topics", "r") as topics_file:
            for line in topics_file.readlines():
                prep_line = map(lambda s: s.strip(), line.split())
                topic_id, doc_id, _ = prep_line
                self.document_topics[doc_id].append(topic_id)

    def __iter__(self, with_topics=False):
        with open(self.path + "/tokens", "r") as tokens_file:
            for idx, line in enumerate(tokens_file.readlines()):
                if idx % 10000 == 0:
                    ln.debug("Corpus iteration at %s..." % idx)
                    if idx and _DEBUG_MODE:
                        break
                if with_topics:
                    tokens = line.split()
                    doc_id = tokens.pop(0)
                    yield tokens, self.document_topics[doc_id]
                else:
                    tokens = line.split()
                    doc_id = tokens.pop(0)
                    yield tokens

    def split_train_test(self, split=0.8, seed_val=1423, with_topics=True):
        ln.info("Splitting data into train and test set using a split of %s" % split)
        seed(seed_val)
        ln.debug("Iterating through corpus and collecting documents.")
        data = list(self.__iter__(with_topics=with_topics))
        ln.info("Read a total of %s documents." % len(data))
        ln.debug("Shuffling data.")
        shuffle(data)
        train = []
        ln.debug("Splitting data into train and test sets.")
        for sample_no in range(int(len(data) * split)):
            train.append(data.pop())
        test = data

        ln.debug("Done.")
        return train, test


corpus = RCV1Corpus("rcv1")

rcv1_train_all, rcv1_test_all = corpus.split_train_test()
ln.debug("Seperating samples from targets..")
rcv1_train, rcv1_train_target = zip(*rcv1_train_all)
rcv1_test, rcv1_test_target = zip(*rcv1_test_all)
ln.debug("Done.")