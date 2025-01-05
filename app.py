from colorthief import ColorThief
import matplotlib.pyplot as plt
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames
feature_list = np.array(pickle.load(open('features_list_for_prods.pkl', 'rb')))
filenames = pickle.load(open('filenames_products.pkl', 'rb'))

# Initialize the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(244, 244, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Configure Streamlit app
# st.set_page_config(
#     page_title="Huematch",
#     layout="wide",
# )

st.title('Huematch')

# Create upload folder if not exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# File upload options
option = st.selectbox('Choose how you want to upload an image', ('Please select', 'Upload image', 'Camera input'))

uploaded_file = None
if option == 'Upload image':
    uploaded_file = st.file_uploader("Choose an image")
elif option == 'Camera input':
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Save uploaded file
    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1
        except:
            return 0

    # Feature extraction function
    def feature_extraction(img_path, model):
        img = image.load_img(img_path, target_size=(244, 244))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result

    # Display color palette function
    def display_color_palette(img_path, num_colors=5):
        color_thief = ColorThief(img_path)
        palette = color_thief.get_palette(color_count=num_colors)
        plt.figure(figsize=(5, 1))
        plt.bar(range(num_colors), [1] * num_colors, color=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in palette], width=1)
        plt.axis('off')
        st.pyplot(plt.gcf())
        plt.close()

    # Recommendation function
    def recommend(features, feature_list, n_recommendations=8):
        neighbors = NearestNeighbors(n_neighbors=n_recommendations + 1, algorithm='brute', metric='cosine')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices, distances

    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        show_original_image = st.checkbox('Show original image alongside recommendations')
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        number_of_recommendations = st.slider('Number of recommendations:', min_value=1, max_value=10, value=5, step=1)
        indices, distances = recommend(features, feature_list, number_of_recommendations)

        columns = st.columns(number_of_recommendations)
        image_width = 550
        image_height = 700

        for i in range(number_of_recommendations):
            if i == 0 and show_original_image:
                display_image = display_image.resize((image_width, image_height))
                columns[i].image(display_image)
            else:
                image_index = i - 1 if show_original_image else i
                if image_index + 1 < len(indices[0]):
                    image_path = filenames[indices[0][image_index + 1]]
                    img = Image.open(image_path)
                    img = img.resize((image_width, image_height))
                    columns[i].image(img)

        if st.button("Show Stats"):
            st.write("Detailed information for the recommended products:")
            for i, distance in enumerate(distances[0][1:1 + number_of_recommendations]):
                with st.expander(f"Product {i + 1}"):
                    st.write(f"Product {i + 1}:")
                    st.write(f"Similarity score: {1 - distance:.4f}")
                    st.write(f"Filename: {filenames[indices[0][i]]}")

                    img = Image.open(filenames[indices[0][i]])
                    img_dimensions = img.size
                    st.write(f"Image dimensions: {img_dimensions}")

                    aspect_ratio = img_dimensions[0] / img_dimensions[1]
                    st.write(f"Aspect ratio: {aspect_ratio:.2f}")

                    st.write(f"Index in feature list: {indices[0][i]}")
                    st.write(f"Raw distance score: {distance:.4f}")
                    st.write("Color palette:")
                    display_color_palette(filenames[indices[0][i]])
    else:
        st.header("Some error occurred during file upload")
