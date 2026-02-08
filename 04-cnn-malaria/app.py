import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Modeli yükle
model = load_model('malaria_cnn_model.h5')

# Resim ön işleme fonksiyonu
def preprocess_image(image):
    img = image.convert('L')  # Grayscale
    img = img.resize((170, 170))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dimension
    img = np.expand_dims(img, axis=0)   # batch dimension
    return img

# Streamlit başlık
st.title("Malaria Cell Image Classification 🦠")
st.write("Bir hücre görüntüsü yükleyin — model bu hücrenin **infected (parasitized)** mi yoksa **uninfected** mi olduğunu tahmin etsin.")

# Resim yükleyici
file = st.file_uploader("Resim yükle", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file)
    st.image(img, caption='Yüklenen Görsel', use_column_width=True)
    
    # Ön işleme
    img_processed = preprocess_image(img)
    
    # Tahmin
    prediction = model.predict(img_processed)
    predicted_class = int(prediction[0][0] > 0.5)
    
    class_names = ['Parasitized', 'Uninfected']
    
    st.write(f"Tahmin: **{class_names[predicted_class]}**")
