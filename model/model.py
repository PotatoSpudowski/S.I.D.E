import efficientnet.tfkeras as efn
import tensorflow_hub as hub
from tensorflow.keras.models import Model

def get_efficientnet_model(model_version):
    if model_version == "B0": return efn.EfficientNetB0(weights='imagenet')
    elif model_version == "B1": return efn.EfficientNetB1(weights='imagenet')
    elif model_version == "B2": return efn.EfficientNetB2(weights='imagenet')
    elif model_version == "B3": return efn.EfficientNetB3(weights='imagenet')
    elif model_version == "B4": return efn.EfficientNetB4(weights='imagenet')
    elif model_version == "B5": return efn.EfficientNetB5(weights='imagenet')
    elif model_version == "B6": return efn.EfficientNetB6(weights='imagenet')
    elif model_version == "B7": return efn.EfficientNetB7(weights='imagenet')
    else: return efn.EfficientNetB0(weights='imagenet')

def get_efficientnet_feature_extractor(model_version):
    effnet_model = get_efficientnet_model(model_version)
    model = Model(inputs=effnet_model.input,
            outputs=effnet_model.get_layer('top_dropout').output)

    return model

def get_sentence_encoder(version):
    urls = {
        "1": "https://tfhub.dev/google/universal-sentence-encoder/1",
        "2": "https://tfhub.dev/google/universal-sentence-encoder/2",
        "3": "https://tfhub.dev/google/universal-sentence-encoder/3",
        "4": "https://tfhub.dev/google/universal-sentence-encoder/4",
        "5": "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    }
    model = hub.load(urls[str(version)])

    return model



