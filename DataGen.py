from DocumentGraph import DocumentGraph
import pickle
from multiprocessing import Pool
import argparse
import gensim

def process_document_set(process_params):
	"""Runs a set of documents through the graph generator and outputs each as a pickle"""
	document_set, index, set_size, model = process_params
	for i in range(0, set_size):
		document = document_set[i]
		document_graph = DocumentGraph(document, model)
		pickle.dump(document_graph, open(output_path + str(index) + ".p", "wb" ))
		index += 1
		print('Completed:', index, 'document')

def partition_documents(documents, partitions, num_documents):
	"""Groups the documents into sets of size num_documents / partitions"""
	document_sets = []
	step = int(num_documents / partitions)
	for i in range(0, num_documents, step):
		document_sets.append(documents[i:i + step])

	return document_sets

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-threads', nargs='?', default=1, type=int, help='Enter number of threads to process the dataset.')
	args = parser.parse_args()

	corpus_path = 'apnews/apnews.dat'
	output_path = 'pickle_data/'
	model_path = 'model/apnews_model.model'
	model = gensim.models.doc2vec.Doc2Vec.load('model/apnews_model.model')

	with open(corpus_path, 'r') as corpus:
		documents = corpus.readlines()
		corpus.close()

	num_documents = len(documents)
	partitions = args.threads
	partition_size = int(num_documents / partitions)
	
	document_sets = partition_documents(documents, partitions, num_documents)

	param_sets = []
	index = 0
	for i in range(partitions):
		param_sets.append((document_sets[i], index, partition_size, model))
		index += partition_size

	p = Pool(args.threads)
	p.map(process_document_set, param_sets)
