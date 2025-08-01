import streamlit as st
import os
import numpy as np
import pickle
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import gdown

# Google Drive download setup
file_id = '1I2XNLFm6fK8n5OU9EFTCTW5OHNgNVqlE'
gdrive_url = f'https://drive.google.com/uc?id={file_id}'

# Download embeddings.pkl from Drive if not present
if not os.path.exists('embeddings.pkl'):
    st.text("Downloading embeddings.pkl from Google Drive...")
    gdown.download(gdrive_url, 'embeddings.pkl', quiet=False)

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Fix Windows paths if needed
def fix_path(path):
    return path.replace('\\', '/')

# Load pretrained model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit app UI
st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader and results
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(fix_path(filenames[indices[0][0]]))
        with col2:
            st.image(fix_path(filenames[indices[0][1]]))
        with col3:
            st.image(fix_path(filenames[indices[0][2]]))
        with col4:
            st.image(fix_path(filenames[indices[0][3]]))
        with col5:
            st.image(fix_path(filenames[indices[0][4]]))
    else:
        st.header("Some error occurred during file upload")
