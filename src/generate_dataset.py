#!/usr/bin/env python

"""
Generate featurized dataset
"""

import sys, os
import xml, argparse
import numpy as np
from corpus_reader import NewsExtractor, GroundTruthExtractor, NewsExtractorFeaturizerFromStream
from document_encoder import DocumentEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('news-dir', help='Hyperpartisan news XML directory')
    parser.add_argument('--ground-truth-dir', help='Ground truth XML directory')
    parser.add_argument('name', help='Name for the generated dataset')

    args = vars(parser.parse_args())
    return args['news-dir'], args['ground_truth_dir'], args['name']


def extract_data(data_dir, data_extractor_constructor):
    data_extractor = data_extractor_constructor()
    for file in os.listdir(data_dir):
        if file.endswith('xml'):
            with open(data_dir + '/' + file, 'r', encoding='utf-8') as in_file:
                parser = xml.sax.make_parser()
                parser.setContentHandler(data_extractor)

                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(in_file)
                source.setEncoding('utf-8')

                parser.parse(source)

    return data_extractor.get_data()


def process_and_group_article_data(news_data_dir, ground_truth_dir):
    """
    Extract articles from news_data_dir, and fill with the truth values
     from ground_truth_dir.
    @returns set of articles.
    """
    articles = extract_data(news_data_dir, NewsExtractor)
    ground_truth = extract_data(ground_truth_dir, GroundTruthExtractor)
    for key, val in ground_truth.items():
        articles[key].set_ground_truth(*val)

    return articles.values()


def process_and_group_data_as_stream(news_data_dir, ground_truth_dir, doc_encoder):
    """
    Process data stream from articles in news_data_dir, and immediately
     featurize articles into data (not needing to keep articles in memory).
    @returns (X, y) featurized dataset
    """
    assert news_data_dir is not None and ground_truth_dir is not None,\
            "news_data_dir or ground_truth_dir not provided"

    ef_constructor = lambda: NewsExtractorFeaturizerFromStream(doc_encoder)
    featurized = extract_data(news_data_dir, ef_constructor)
    ground_truth = extract_data(ground_truth_dir, GroundTruthExtractor)

    print('Moving data to ndarray...')
    X, y = None, None
    for idx, (key, val) in enumerate(featurized.items()):
        if X is None:
            X = np.ndarray((len(featurized), *(val.shape)), dtype=val.dtype)
            y = np.ndarray((len(ground_truth),), dtype=np.bool)

        X[idx] = val
        y[idx] = (ground_truth[key][0] == 'true')

    return X, y


def extract_texts(news_data_dir):
    ef_constructor = lambda: NewsExtractorFeaturizerFromStream(lambda x: x.get_text_cleaned())
    return extract_data(news_data_dir, ef_constructor)


def generate_featurized_dataset(news_data_dir, ground_truth_dir):
    articles = process_and_group_article_data(news_data_dir, ground_truth_dir) \
            if ground_truth_dir is not None else extract_data(news_data_dir, NewsExtractor).values()

    ## Transform data
    print('Featurizing articles...')
    encoder = DocumentEncoder(articles)
    X = encoder.featurize(articles)
    y = np.fromiter(map(lambda a: a.get_hyperpartisan() == 'true', articles), dtype=np.bool) \
        if ground_truth_dir is not None else None

    return X, y


if __name__ == '__main__':
    import numpy as np
    import pickle
    
    news_data_dir, ground_truth_dir, name = parse_args()

    data = extract_texts(news_data_dir)
    doc_encoder = DocumentEncoder(data.values())

    X, y = process_and_group_data_as_stream(news_data_dir, ground_truth_dir, doc_encoder)

    ## Saving DocumentEncoder
    doc_encoder.texts = None ## Delete texts so as to not save them in the pickle...
    doc_encoder_path = '../generated_datasets/DocEncoder_{}.pickle'.format(name)
    print('Savind DocumentEncoder to "{}"'.format(doc_encoder_path))
    pickle.dump(doc_encoder, open(doc_encoder_path, 'wb'))

    ## Saving Dataset
    dataset_path = '../generated_datasets/{}.npz'.format(name)
    print('Saving generated dataset to "{}"'.format(dataset_path))
    np.savez_compressed(dataset_path, X=X, y=y)
