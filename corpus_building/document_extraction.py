import os
import nltk
from nltk.corpus import stopwords

def extract_all_text_and_store(file_dir, out_dir):
    '''Extracts all of the sentences from all of the files in the 
    source directory and stores the sentences in a file corresponding 
    to each source file.
    '''
    all_sentences = []

    directory = os.fsencode(file_dir)
    for source_file in os.listdir(directory):
        filename = os.fsdecode(source_file)
        file_path = file_dir + filename
        out_path = out_dir + filename

        sentences = _extract_text_from_file(file_path)
        
        with open(out_path, 'w') as out_file:
            for sentence in sentences:
                out_file.write(sentence + '\n')
            out_file.close()

    return all_sentences

def _extract_text_from_file(file_path):
    '''Returns a list of all of the sentences from all of the bodies of text in the file.'''
    return _extract_text_from_file_helper(open(file_path, 'r').readlines())

def _extract_text_from_file_helper(all_lines):
    '''Extracts all of the sentences in the <TEXT> ... </TEXT> section of an apnews document'''
    START_FLAG = '<TEXT>'
    END_FLAG = '</TEXT>'
    sentences = []
    current_doc = ""
    in_text = False

    for line in all_lines:
        if START_FLAG in line:
            in_text = True
            continue
        elif END_FLAG in line:
            in_text = False
            sentences += _remove_stopwords_and_clean_sentences(nltk.sent_tokenize(current_doc))
            current_doc = ""
            continue

        if in_text:
            current_doc += line.replace('\n', ' ')

    return sentences

def _remove_stopwords_and_clean_sentences(sentences):
    '''Parameter sentences is a list of strings.
    Returns a list of lists, where each list is a list of words in the sentence.
    '''
    cleaned_sentences = []
    english_stopwords = set(stopwords.words('english'))

    for sentence in sentences:
        cleaned_sentences.append(' '.join([word for word in sentence.lower().split() if word not in english_stopwords]))

    return cleaned_sentences
        

        
