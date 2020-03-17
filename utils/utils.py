import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm
from tensorflow.keras.models import Model
from efficientnet.keras import center_crop_and_resize, preprocess_input

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True

def preprocess_image(image,image_size):
    image = io.imread(image)
    x = center_crop_and_resize(image, image_size=image_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    return x

def extract_image_feature(image, model):
    return model.predict(image)[0]

def extract_image_features(images, model):
    features = []
    for image in tqdm(images):
        features.append(extract_image_feature(preprocess_image(image, model.input_shape[1]), model))
    
    return features

def generate_sentence_embedding(sentence, model):
    return model([sentence])[0]

def generate_sentence_embeddings(sentences, model):
    batch_size = 50000
    if len(sentences) <= batch_size:
        return model(sentences)
    else:
        embeddings = []
        for i in tqdm(range(len(sentences)//batch_size)):
            batch = sentences[i*batch_size:(i+1)*batch_size]
            embeddings.extend(model(batch))
        if len(sentences)%batch_size != 0:
            batch = sentences[(i+1)*batch_size:]
        
        return embeddings 

def clean_data(df_path, ignore_df_path):
    df = pd.read_csv(df_path)
    ignore_df = pd.read_csv(ignore_df_path)
    ignore_df.columns = ['index', 'images']
    common = pd.merge(df, ignore_df, on=['images'])
    s1 = df[(~df.images.isin(common.images))]

    return s1   

def generate_image_embedding_pair(df, model):
    images, captions = df["images"], df["captions"]
    embeddings = generate_sentence_embeddings(captions, model)

    return list(zip(images, embeddings))

