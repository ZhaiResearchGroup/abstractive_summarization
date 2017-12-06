import sys
sys.path.append("..") # allows for cross-directory imports

from corpus_building import dataset_loader
from model_train import train_model, get_new_model

import os
import gensim

if __name__ == "__main__":

	chunk_size = 500
	model_dir = '../model/'
	dataset_dir = '../corpus/'
	sentences = dataset_loader.load_all_sentences(dataset_dir)
	training_corpus = [gensim.models.doc2vec.TaggedDocument(sen, [i]) for (i, sen) in enumerate(sentences)]

	print('Data Read.')

	if os.listdir(model_dir) == []:
		print('No existing model found. Using new model.')
		model = get_new_model(training_corpus)
	else:
		print('Using existing model.')
		model_path = model_dir + os.listdir(model_dir)[0]
		model = gensim.models.doc2vec.Doc2Vec.load(model_path)

	print('Beginning training.')

	train_model(model, training_corpus, chunk_size)

	print('Training Finished.')

	model.save('../model/apnews_sen_model.model')

	print('Model Saved.')
