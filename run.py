from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import nltk.tokenize
import pprint
pp = pprint.PrettyPrinter(indent=2)
from random import shuffle
from sklearn.linear_model import LogisticRegression


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('bad prefix key')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


files = {'train_pos.txt':'TRAIN_POS','train_neg.txt':'TRAIN_NEG','test_pos.txt':'TEST_POS','test_neg.txt':'TEST_NEG'}
sents = LabeledLineSentence(files)
model = Doc2Vec(min_count=1, window=10, size=15, sample=1e-4, negative=5, workers=8)
model.build_vocab(sents.to_array())

# training
model.train(sents.sentences_perm(),total_examples=166553,epochs=model.iter)
model.save('./reviews.d2v')

model = Doc2Vec.load('./reviews.d2v')
pp.pprint(model.most_similar('bad'))

pp.pprint(model.most_similar('good'))
print model.docvecs['TRAIN_POS_0']
# train_pos 119381
train_pos_size = 13860 # this is to equalize the datasets
# train_pos_size = 119381
train_neg_size = 13860
train_size = train_pos_size + train_neg_size
train_arrays = np.zeros((train_size,15))
train_labels = np.zeros(train_size)
for i in range(train_pos_size):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(train_neg_size):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i + train_pos_size] = model.docvecs[prefix_train_neg]
    train_labels[i + train_pos_size] = 0

# total train: 133241


# test_pos 29846
# test_neg 3466
# total test: 33312
test_pos_size = 3466 # equalize
# test_pos_size = 29846
test_neg_size = 3466
test_size = test_pos_size + test_neg_size 

test_arrays = np.zeros((test_size,15))
test_labels = np.zeros(test_size)
for i in range(test_pos_size):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1
for i in range(test_neg_size):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i + test_pos_size] = model.docvecs[prefix_test_neg]
    test_labels[i + test_neg_size] = 0

classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print classifier.score(test_arrays, test_labels)
