# Intel Image Classification Dataseti ile Landscape tahmini

Bu proje, Intel Image Classification veri setini kullanarak görüntü sınıflandırma yapan bir derin öğrenme modeli ve kullanıcı dostu bir web arayüzü içerir.

## Proje Hakkında

Proje, 6 farklı doğal manzara kategorisini (buildings, forest, glacier, mountain, sea, street) sınıflandıran bir Convolutional Neural Network (CNN) modeli içerir. Model, veri artırma teknikleri kullanılarak eğitilmiş ve yüksek doğruluk oranına sahiptir (%90). Eğitilen model, kullanıcıların kendi görüntülerini yükleyip sınıflandırma yapabilecekleri bir web arayüzü ile sunulmaktadır.

## Veri Seti

Intel Image Classification veri seti şu kategorileri içerir:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Veri seti eğitim ve test olmak üzere iki alt kümeden oluşmaktadır.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri setini indirin ve aşağıdaki dizin yapısını oluşturun:
```
intel_image_classification/
├── seg_train/
│   └── seg_train/
└── seg_test/
    └── seg_test/
```

## Kullanım

### Model Eğitimi

Modeli eğitmek için:
```bash
python train_model.py
```

Bu komut:
- Veri setini yükleyecek ve ön işleyecek
- Veri artırma teknikleri uygulayacak
- Modeli eğitecek (20 epoch)
- Early stopping ve learning rate reduction kullanacak
- Eğitim geçmişini görselleştirecek
- En iyi modeli kaydedecek
- Test seti üzerinde değerlendirme yapacak

### Web Arayüzü

Web arayüzünü başlatmak için:
```bash
streamlit run app.py
```

Arayüz şu özellikleri sunar:
- Görüntü yükleme ve önizleme
- Tahmin sonuçlarını görselleştirme
- Her sınıf için olasılık dağılımı
- Detaylı sonuç raporu
- Eğitilmiş model ile gerçek zamanlı sınıflandırma

## Model Mimarisi

Model şu katmanları içerir:
- 4 adet konvolüsyonel blok (her blokta Conv2D, BatchNormalization, MaxPooling2D ve Dropout)
- Global Average Pooling
- 3 adet tam bağlantılı katman (Dense, BatchNormalization, Dropout)

## Veri Artırma Teknikleri

Eğitim sırasında şu veri artırma teknikleri uygulanır:
- Döndürme (20 derece)
- Yatay ve dikey kaydırma (%20)
- Kesme (shear) (%20)
- Yakınlaştırma (%20)
- Yatay çevirme
- Normalizasyon (1./255)

## Gereksinimler

- Python 3.7+
- TensorFlow 2.8.0+
- Streamlit 1.22.0+
- Matplotlib 3.5.0+
- NumPy 1.21.0+
- Pillow 9.0.0+


## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Dalınıza push yapın (`git push origin yeni-ozellik`)
5. Bir Pull Request oluşturun

## İletişim

Sorularınız veya önerileriniz için lütfen bir Issue açın. 