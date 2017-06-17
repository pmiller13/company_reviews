import glob
import json
import pprint
import string
printable = set(string.printable)
punc = set(string.punctuation)
pp = pprint.PrettyPrinter(indent=2)
pos = []
neg = []
pos_temp = json.load(open('./positive/693_webhose-2017-03_20170404094005/discussions_0000011.json'))
pp.pprint(pos_temp)


for f in glob.glob('./positive/693_webhose-2017-02_20170404094642/*.json'):
    pos.append(json.load(open(f)))
for f in glob.glob('./positive/693_webhose-2017-03_20170404094005/*.json'):
    pos.append(json.load(open(f)))

for f in glob.glob('./negative/696_webhose-2017-02_20170404095409/*.json'):
    neg.append(json.load(open(f)))
for f in glob.glob('./negative/696_webhose-2017-03_20170404095235/*.json'):
    neg.append(json.load(open(f)))


# neg contains all negative reviews

# Filter non-english reviews and log language stats
languages = {}
pos_engl = []
neg_engl = []

for review in neg:
    if review['language'] in languages:
        languages[review['language']] = languages[review['language']] + 1
    else:
        languages[review['language']] = 1
    if review['language'] == 'english':
        neg_engl.append(review['text'])
print(str(len(neg_engl)) + " total english negative reviews")

for review in pos:
    if review['language'] in languages:
        languages[review['language']] = languages[review['language']] + 1
    else:
        languages[review['language']] = 1
    if review['language'] == 'english':
        pos_engl.append(review['text'])
print(str(len(pos_engl)) + " total english positive reviews")

print("Num positive: " + str(len(pos_engl)))

print("Num negative: " + str(len(neg_engl)))
print("English total: " + str(languages['english']))
print("Writing to files...")
#print pos_engl[0]

pos_out = open('pos.txt','w')
neg_out = open('neg.txt','w')

for out in pos_engl:
    out = filter(lambda x: x in printable, out)
    out = filter(lambda x: x not in punc, out)
    pos_out.write("%s\n" % out)
for out in neg_engl:
    out = filter(lambda x: x in printable, out)
    out = filter(lambda x: x not in punc, out)
    neg_out.write("%s\n" % out)

pos_out.close()
neg_out.close()

# split training/testing
positive = [line.rstrip('\n') for line in open('pos.txt')]
negative = [line.rstrip('\n') for line in open('neg.txt')]

print("Splitting 80/20 training/testing")
pos_bar = int(len(positive) * .8)
print pos_bar
neg_bar = int(len(negative) * .8)

pos_train = positive[:pos_bar]
pos_out = open('train_pos.txt', 'w')
for out in pos_train:
    pos_out.write("%s\n" % out)
pos_out.close()

pos_test = positive[pos_bar:]
pos_out = open('test_pos.txt', 'w')
for out in pos_test:
    pos_out.write("%s\n" % out)
pos_out.close()

neg_train = negative[:neg_bar]
neg_out = open('train_neg.txt', 'w')
for out in neg_train:
    neg_out.write("%s\n" % out)
neg_out.close()

neg_test = negative[neg_bar:]
neg_out = open('test_neg.txt', 'w')
for out in neg_test:
    neg_out.write("%s\n" % out)
neg_out.close()
