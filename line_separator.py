import tokenizer


corpus_in_path = 'apnews/apnews.dat'
corpus_out_path = 'apnews/apnews_sen.dat'

with open(corpus_in_path, 'r') as corpus:
    documents = corpus.readlines()
    corpus.close()

N_docs = float(len(documents))

sentences = list()
for i, doc in enumerate(documents):
    sentences.extend(tokenizer.tokenize_text(doc))

    if i % 10000 == 0:
        print i, "out of", N_docs, "(" + str(100 * i / N_docs) + "%)"

text = '\n'.join(sentences)

with open(corpus_out_path, 'w') as corpus:
    corpus.write(text)
