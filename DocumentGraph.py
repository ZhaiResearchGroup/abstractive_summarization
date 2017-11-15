import gensim
from nltk import tokenize
import collections

DEFAULT_VEC_SIZE = 100
DEFAULT_MIN_WORD_COUNT = 1
DEFAULT_TRAINING_ITERATIONS = 100

class DocumentGraph:

	def __init__(self, document, vec_size = DEFAULT_VEC_SIZE, min_word_count = DEFAULT_MIN_WORD_COUNT, training_iterations = DEFAULT_TRAINING_ITERATIONS):
		self.base_document = document
		self.training_parameters = (vec_size, min_word_count, training_iterations)
		self.similarity_matrix, self.sentence_ids = self.build_matrix(document)

	def build_matrix(self, document):
		"""Returns a NxN matrix where the rows and columns are the sentence embeddings
		of the sentences in the document and the i,j entry is the similarity between 
		sentence embedding i and sentence embedding j
		"""
		sentences = tokenize.sent_tokenize(document)
		num_sentences = len(sentences)
		
		training_corpus, sentence_ids = self._build_corpus(sentences, num_sentences)
		similarity_matrix = self._init_similarity_matrix(num_sentences)

		if num_sentences < 1:
			return similarity_matrix, sentence_ids

		trained_model = self._train_model(training_corpus)

		similarity_matrix = self._update_similarity_matrix(trained_model, training_corpus, similarity_matrix)

		return similarity_matrix, sentence_ids

	def _init_similarity_matrix(self, dim):
		"""Returns a 2D matrix of a specified dimension where each i,j value is -1"""
		return [[-1 for i in range(dim)] for j in range(dim)]

	def _build_corpus(self, sentences, num_sentences):
		"""Returns...
		sentence_ids: a map where the key is the id of the sentence and the value is the tagged document object of the sentence
		training_corpus: a list of tagged document objects where each object is a sentence
		"""
		sentence_ids = {}
		training_corpus = []

		for i in range(num_sentences):
			tagged_document = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(sentences[i]), [i])
			training_corpus.append(tagged_document)
			sentence_ids[i] = tagged_document

		return training_corpus, sentence_ids

	def _train_model(self, training_corpus):
		"""Returns a trained model given the training parameters and a corpus"""
		size, min_count, iterations = self.training_parameters
		model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=min_count, iter=iterations)

		model.build_vocab(training_corpus)

		model.train(training_corpus, total_examples=model.corpus_count, epochs=model.iter)

		return model

	def _update_similarity_matrix(self, trained_model, training_corpus, similarity_matrix):
		"""Updates a similarity matrix with the similarities between each of the sentences and all of the other sentences.
		The i,j value contains the similarity between sentence i and sentence j.
		"""
		for doc_id in range(len(training_corpus)):
			inferred_vector = trained_model.infer_vector(training_corpus[doc_id].words)
			sims = trained_model.docvecs.most_similar([inferred_vector], topn=len(trained_model.docvecs))

			for sim in sims:
				similarity_matrix[doc_id][sim[0]] = sim[1]


		return similarity_matrix


