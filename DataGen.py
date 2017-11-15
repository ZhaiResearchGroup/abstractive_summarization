from DocumentGraph import DocumentGraph
import pickle
from multiprocessing import Process

def process_document_set(document_set, start, set_size):
	for i in range(start, start + set_size):
		document = document_set[i]
		document_graph = DocumentGraph(document)
		pickle.dump(document_graph, open(output_path + str(i) + ".p", "wb" ))
		print('Completed:', i, 'document')

if __name__ == "__main__":

	corpus_path = 'apnews/apnews.dat'
	output_path = 'pickle_data/'

	with open(corpus_path, 'r') as corpus:
		documents = corpus.readlines()
		corpus.close()

	num_documents = len(documents)
	
	first_document_set = documents[0:int(num_documents/3)]
	second_document_set = documents[int(num_documents/3):int(num_documents/3+num_documents/3)]
	third_document_set = documents[int(num_documents/3+num_documents/3):]

	first_set_process = Process(target=process_document_set, args=(first_document_set, 0, int(num_documents/3)))
	first_set_process.start()
	first_set_process.join()


	second_set_process = Process(target=process_document_set, args=(second_document_set, int(num_documents/3), int(num_documents/3 + num_documents/3)))
	second_set_process.start()
	second_set_process.join()

	third_set_process = Process(target=process_document_set, args=(third_document_set, int(num_documents/3+num_documents/3), num_documents))
	third_set_process.start()
	third_set_process.join()

