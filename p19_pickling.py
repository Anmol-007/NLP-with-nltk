import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode 


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
	
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		
		choice_votes = votes.count(mode(votes))
		conf = float(choice_votes) / len(votes)
		return conf
		
		
		
short_pos = open("positive.txt", 'r').read()
short_neg = open("negative.txt", 'r').read()

documents = []
all_words = []
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
	documents.append( (p, "pos") )
	words = word_tokenize(p.decode('iso-8859-1'))
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())
			
for p in short_neg.split('\n'):
	documents.append( (p, "neg") )
	words = word_tokenize(p.decode('iso-8859-1'))
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())


save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

	
all_words = nltk.FreqDist(all_words)

#print len(all_words.keys())
word_features = list(all_words.keys())[:5000]; #use only top 5000 most common words

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
	words = word_tokenize(document.decode('iso-8859-1'))
	features = {}
	for w in word_features:
		features[w] = (w in words) # returns boolean 
	
	return features
	
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_word_featuresets = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_word_featuresets)
save_word_featuresets.close()

#print(len(documents))


random.shuffle(featuresets)
#positive reviews data
training_set = featuresets[:10000]
testing_set = featuresets[10000:]	

#negative reviews data
#training_set = featuresets[100:]
#testing_set = featuresets[:100]	


#Naive Bayes
# posterior = prior_occurances * likelihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)


print('Original Naive Bayes Algo accuracy percent:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

originalNB_classifier = open("pickled_algos/original_naivebayes.pickle", "wb")
pickle.dump(classifier, originalNB_classifier)
originalNB_classifier.close()


#MultinomialNB from scikitlearn
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MultinomialNB Algo accuracy percent:', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

MNB_classifier_pickle = open("pickled_algos/MNB_naivebayes.pickle", "wb")
pickle.dump(MNB_classifier, MNB_classifier_pickle)
MNB_classifier_pickle.close()


#GaussianNB
#GNB_classifier = SklearnClassifier(GaussianNB())
#GNB_classifier.train(training_set)
#print('GaussianNB Algo accuracy percent:', (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

#BernoulliNB
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print('BernoulliNB Algo accuracy percent:', (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

BNB_classifier_pickle = open("pickled_algos/BNB_naivebayes.pickle", "wb")
pickle.dump(BNB_classifier, BNB_classifier_pickle)
BNB_classifier_pickle.close()

#LogisticRegression, SDGClassifier
#SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression Algo accuracy percent:', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

LogisticRegression_classifier_pickle = open("pickled_algos/LogisticRegression_naivebayes.pickle", "wb")
pickle.dump(LogisticRegression_classifier, LogisticRegression_classifier_pickle)
LogisticRegression_classifier_pickle.close()


#SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#SGDClassifier_classifier.train(training_set)
#print('SGDClassifier Algo accuracy percent:', (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC Algo accuracy percent:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

LinearSVC_classifier_pickle = open("pickled_algos/LinearSVC_naivebayes.pickle", "wb")
pickle.dump(LinearSVC_classifier, LinearSVC_classifier_pickle)
LinearSVC_classifier_pickle.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC Algo accuracy percent:', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

NuSVC_classifier_pickle = open("pickled_algos/NuSVC_naivebayes.pickle", "wb")
pickle.dump(NuSVC_classifier, NuSVC_classifier_pickle)
NuSVC_classifier_pickle.close()

## Vote Classifier & Confidence 

## 7 classfiers
voted_classifier =  VoteClassifier(	MNB_classifier, 
									BNB_classifier, 
									LogisticRegression_classifier, 
									LinearSVC_classifier, 
									NuSVC_classifier)



print('voted_clasifier Algo accuracy percent:', (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
	feats = find_features(text)
	return voted_classifier.classify(feats)

#print (testing_set[0][0])


