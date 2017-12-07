import gensim
from nltk import tokenize
import tokenizer
import sys

DEFAULT_VEC_SIZE = 100
DEFAULT_MIN_WORD_COUNT = 1
DEFAULT_TRAINING_ITERATIONS = 100

def train_model(model, training_corpus, chunk_size, starting_index):
    """Returns a trained model given the training parameters and a corpus."""
    corpus_size = len(training_corpus)
    safe_model = model

    for i in range(starting_index, corpus_size, chunk_size):
        try:
            safe_model = model.train(training_corpus[i:i+chunk_size], total_examples=chunk_size, epochs=model.iter)
            print(i + chunk_size, 'sentences trained on.')
        except KeyboardInterrupt:
            print('Finishing current iteration.')
            safe_model = model.train(training_corpus[i:i+chunk_size], total_examples=chunk_size, epochs=model.iter)
            print('Iteration finished. Saving model.')
            model.save('../model/apnews_sen_model.model')

            with open('../model/current_document.txt', 'w') as index_file:
                index_file.write(str(i + chunk_size))
                index_file.close()

            print('Model saved.')
            sys.exit(0)

    return model

def get_new_model(training_corpus, training_parameters=(DEFAULT_VEC_SIZE, DEFAULT_MIN_WORD_COUNT, DEFAULT_TRAINING_ITERATIONS)):
    '''Returns a new model ready to be trained.'''
    size, min_count, iterations = training_parameters
    model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=min_count, iter=iterations)

    model.build_vocab(training_corpus)

    return model