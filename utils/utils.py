import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from efficientnet.keras import center_crop_and_resize, preprocess_input

class CustomGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, image_filenames, embeddings, batch_size, sentence_encoder_model):
        self.image_filenames, self.embeddings = image_filenames, embeddings
        self.batch_size = batch_size
        self.sentence_encoder_model = sentence_encoder_model

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[int(idx * self.batch_size):int((idx + 1) * self.batch_size)]
        batch_y = self.embeddings[int(idx * self.batch_size):int((idx + 1) * self.batch_size)]
        embeddings = self.sentence_encoder_model(batch_y)
  
        images = []
        embeddings2 = []
        for i in range(len(batch_x)):
            try:
                if verify_image(os.path.join("Data", batch_x[i])):
                    image = preprocess_image(os.path.join("Data", batch_x[i]), 380)
                    if image.shape == (1,380,380,3):
                        images.append(image[0])
                        embeddings2.append(embeddings[i])
            except ValueError:
                pass

        return np.asarray(images), np.asarray(embeddings2)

def verify_image(img_file):
    try:
        img = io.imread(img_file)
        if img.shape == (0, 0) or img.shape == (1, 1):
            return False
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
    embeddings2 = []
    for embedding in tqdm(embeddings):
        embeddings2.append(tf.convert_to_tensor(np.reshape(embedding, (1, 512)), np.float32))

    return list(zip(images, embeddings2))

