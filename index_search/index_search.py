import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'utils')))

from annoy import AnnoyIndex
from tqdm import tqdm
from utils.utils import *

def build_annoy_index(features, feature_size, no_of_trees):
    if type(features) != np.ndarray:
        features = np.asarray(features)
    annoy_index = AnnoyIndex(feature_size, metric='angular')
    for index, vector in tqdm(enumerate(features)):
        annoy_index.add_item(index, vector)
    annoy_index.build(no_of_trees)

    return annoy_index

def nns_for_images(image, model, annoy_index, no_of_nns=5, include_distances=True):
    feature = extract_image_feature(preprocess_image(image, model.input_shape[1]), model)
    distances = annoy_index.get_nns_by_vector(feature, no_of_nns, include_distances=True)

    return distances

def nns_for_sentence(sentence, model, annoy_index, no_of_nns=5, include_distances=True):
    feature = generate_sentence_embedding(sentence, model)
    distances = annoy_index.get_nns_by_vector(feature, no_of_nns, include_distances=True)

    return distances

