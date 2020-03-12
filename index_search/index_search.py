import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'model')))

from annoy import AnnoyIndex
from tqdm import tqdm
from model import model

def build_annoy_indexer(features, feature_size, no_of_trees):
    annoy_indexer = AnnoyIndex(feature_size, metric='angular')
    for index, vector in tqdm(enumerate(features)):
        annoy_indexer.add_item(index, vector)
    annoy_indexer.build(no_of_trees)

    return annoy_indexer

def nns_for_images(image, model, annoy_indexer, no_of_nns=5, include_distances=True):
    feature = model.extract_image_feature(model.preprocess_image(image, model.input_shape[1]), model)
    distances = annoy_indexer.get_nns_by_vector(feature, no_of_nns, include_distances=True)

    return distances

