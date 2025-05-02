import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="📷",
    layout="centered"
)

# Başlık ve açıklama
st.title("Intel Image Classification")
st.write("Bu uygulama, yüklediğiniz görüntüleri 6 farklı kategoride sınıflandırır: buildings, forest, glacier, mountain, sea ve street.")

# Sınıf isimleri
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Modeli yükleme
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.keras')
        st.success("Model başarıyla yüklendi!")
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        st.info("Lütfen önce modeli eğitin (train_model.py çalıştırın)")
        return None

# Görüntü ön işleme
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Tahmin fonksiyonu
def make_prediction(img_array, model):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence, predictions[0]

# Ana uygulama
def main():
    # Model yükleme
    model = load_model()
    
    if model is None:
        return
    
    # Dosya yükleme
    uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Görüntüyü yükle ve göster
        img = Image.open(uploaded_file)
        st.image(img, caption="Yüklenen Görüntü", use_column_width=True)
        
        # Görüntüyü ön işle
        img_array = preprocess_image(img)
        
        # Tahmin yap
        predicted_class, confidence, all_predictions = make_prediction(img_array, model)
        
        # Sonuçları göster
        st.subheader("Tahmin Sonuçları")
        
        # En yüksek olasılıklı sınıf
        st.write(f"**Tahmin Edilen Sınıf:** {class_names[predicted_class]}")
        st.write(f"**Güven Skoru:** {confidence:.2%}")
        
        # Tüm sınıfların olasılıklarını göster
        st.subheader("Tüm Sınıfların Olasılıkları")
        
        # Görselleştirme
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(class_names, all_predictions)
        
        # En yüksek olasılıklı sınıfı vurgula
        bars[predicted_class].set_color('red')
        
        plt.xticks(rotation=45)
        plt.ylabel('Olasılık')
        plt.title('Sınıf Olasılıkları')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Olasılıkları tablo olarak göster
        st.subheader("Detaylı Olasılıklar")
        for i, (class_name, prob) in enumerate(zip(class_names, all_predictions)):
            st.write(f"{class_name}: {prob:.2%}")

if __name__ == "__main__":
    main() 