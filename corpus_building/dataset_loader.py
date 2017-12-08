import os

def load_all_sentences(source_dir):
    '''Returns a list of all of the sentences from the post-processed files in the apnews dataset'''
    all_sentences = []

    directory = os.fsencode(source_dir)
    for data_file in os.listdir(directory):
        filename = os.fsdecode(data_file)

        file_path = source_dir + filename
        all_sentences += load_sentences_from_file(file_path)

    return all_sentences

def load_sentences_from_file(file_path):
    '''Returns a list of the sentences from a single post-processed file in the apnews dataset'''
    sentences = open(file_path, 'r').readlines()
    return [sentence.replace('\n', ' ').split() for sentence in sentences]
