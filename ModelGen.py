import gensim
from nltk import tokenize

DEFAULT_VEC_SIZE = 100
DEFAULT_MIN_WORD_COUNT = 1
DEFAULT_TRAINING_ITERATIONS = 100

def train_model(training_corpus, training_parameters):
		"""Returns a trained model given the training parameters and a corpus"""
		size, min_count, iterations = training_parameters
		model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=min_count, iter=iterations)

		model.build_vocab(training_corpus)

		model.train(training_corpus, total_examples=model.corpus_count, epochs=model.iter)

		return model

if __name__ == "__main__":

	dataset_path = 'apnews/apnews.dat'
	training_corpus = []

	with open(dataset_path, 'r') as dataset:
		documents = dataset.readlines()
		dataset.close()

	print('Data Read.')

	sentence_count = 0
	document_count = 0 # temp for testing
	for document in documents:
		sentences = tokenize.sent_tokenize(document)
		
		for sentence in sentences:
			tagged_sentence = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(sentence), [s_i])
			training_corpus.append(tagged_sentence)
			sentence_count += 1

		document_count += 1 # temp for testing
		if document_count > 20:
			break

	print('Corpus Generated.')

	training_parameters = DEFAULT_VEC_SIZE, DEFAULT_MIN_WORD_COUNT, DEFAULT_TRAINING_ITERATIONS
	model = train_model(training_corpus, training_parameters)

	print('Training Finished.')

	model.save('model/apnews_model.model')

	print('Model Saved.')





