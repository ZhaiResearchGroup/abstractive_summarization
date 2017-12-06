from document_extraction import extract_all_text_and_store
from dataset_loader import load_sentences_from_file, load_all_sentences

if __name__ == "__main__":

    source_dir = '../apnews_source_dataset/'
    out_dir = '../corpus/'

    extract_all_text_and_store(source_dir, out_dir)