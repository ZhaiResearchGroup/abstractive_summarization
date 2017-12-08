import sys
sys.path.append("..") # allows for cross-directory imports

from corpus_building import dataset_loader
from model_train import train_model, get_new_model

import os
import gensim

def get_file_in_dir(file_list, filename_keyword):
	for filename in file_list:
		if filename_keyword in filename:
			return filename
	return None

if __name__ == "__main__":

	chunk_size = 500
	model_dir = '../model/'
	dataset_dir = '../corpus/'
	sentences = dataset_loader.load_all_sentences(dataset_dir)
	training_corpus = [gensim.models.doc2vec.TaggedDocument(sen, [i]) for (i, sen) in enumerate(sentences)]

	print('Data Read.')

	index = 0
	fully_trained = False

	filenames = os.listdir(model_dir)
	model_filename = get_file_in_dir(filenames, '.model')
	index_filename = get_file_in_dir(filenames, '.txt')

	model_path = (model_dir + model_filename) if model_filename is not None else None
	index_path = (model_dir + index_filename) if index_filename is not None else None

	if os.listdir(model_dir) == []:
		print('No existing model found. Using new model.')
		model = get_new_model(training_corpus)
	elif len(filenames) == 1:
		print('Model is fully trained on the apnews dataset. No more training will occur.') 
		fully_trained = True
	else:
		print('Using existing model.')
		index = int(open(index_path, 'r').readlines()[0].replace('\n', ''))
		model = gensim.models.doc2vec.Doc2Vec.load(model_path)

	if not fully_trained:
		print('Beginning training.')

		train_model(model, training_corpus, chunk_size, index)

		print('Training Finished.')

		if index_path is not None:
			os.remove(index_path)

		model.save('../model/apnews_sen_model.model')

		print('Model Saved.')
