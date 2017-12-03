import gensim
from nltk import tokenize
import collections
import numpy as np

DEFAULT_VEC_SIZE = 100
DEFAULT_MIN_WORD_COUNT = 1
DEFAULT_TRAINING_ITERATIONS = 100

class DocumentGraph:

	def __init__(self, sentences, trained_model, vec_size = DEFAULT_VEC_SIZE, min_word_count = DEFAULT_MIN_WORD_COUNT, training_iterations = DEFAULT_TRAINING_ITERATIONS):
		self.base_document = ' '.join([' '.join(sen) + '.' for sen in sentences])
		self.training_parameters = (vec_size, min_word_count, training_iterations)
		self.similarity_matrix, self.sentence_ids = self.build_matrix(sentences, trained_model)

	def build_matrix(self, sentences, trained_model):
		"""Returns a NxN matrix where the rows and columns are the sentence embeddings
		of the sentences in the document and the i,j entry is the similarity between
		sentence embedding i and sentence embedding j
		"""
		# sentences = tokenize.sent_tokenize(document)
		num_sentences = len(sentences)

		sentence_corpus, sentence_ids = self._build_corpus(sentences, num_sentences)
		similarity_matrix = self._init_similarity_matrix(num_sentences)

		if num_sentences < 1:
			return similarity_matrix, sentence_ids

		similarity_matrix = self._create_similarity_matrix(trained_model, sentence_corpus, similarity_matrix)

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
		sentence_corpus = []

		for i in range(num_sentences):
			sentence_corpus.append((i, sentences[i]))
			sentence_ids[i] = sentences[i]

		return sentence_corpus, sentence_ids

	def n_similarity_unseen_docs(self, model, ds1, ds2):
		"""
		Compute cosine similarity between two sets of docvecs of out-of-training-set docs, as if from inference
		"""

		v1 = [get_infered_vec(doc) for doc in ds1]
		v2 = [get_infered_vec(doc) for doc in ds2]
		return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

	def get_infered_vec(model, doc, alpha=0.1, min_alpha=0.0001, steps=5):
		doc_words = gensim.utils.simple_preprocess(doc)
		return model.infer_vector(doc_words=doc_words, alpha=alpha, min_alpha=min_alpha, steps=steps)

	def _create_similarity_matrix(self, trained_model, sentence_corpus, similarity_matrix):
		"""Updates a similarity matrix with the similarities between each of the sentences and all of the other sentences.
		The i,j value contains the similarity between sentence i and sentence j.
		"""
		for sentence_id, sentence in sentence_corpus:
			for compare_sentence_id, compare_sentence in sentence_corpus:
				similarity = trained_model.docvecs.similarity_unseen_docs(model= trained_model, doc_words1 = sentence, doc_words2 = compare_sentence)

				similarity_matrix[sentence_id][compare_sentence_id] = similarity

		return np.array(similarity_matrix)
