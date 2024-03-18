import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


(X_train,y_train),(X_test,y_test) = datasets.cifar10.load_data()    # dataseti importladık
X_train.shape         # 50 bin tanesi train için kullanılacak, toplam resim 60 bin, 32x32 piksel ve 3 kanal RGB'den olusuyo
X_test.shape

y_train[:3] # ilk 3 indeksine bakıyoruz

# y_train ve y_test 2 boyutlu array olarak tutuluyor cifar10 verisetinde. Biz bu verileri görsel olarak daha rahat anlamak için tek boyutlu hale getiriyoruz.
# 2 boyutlu arrayi tek boyuta getirmek için reshape() kullanıyoruz. 

y_test = y_test.reshape(-1,) # 1 boyut eksilttik 
y_test 

#%%  Verilere göz atalım. Kendimiz bir array olusturucaz

resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]  # datasetteki aynı sırayla array olusturucaz


def plot_sample(X,y,index):        # indexe göre label'ını vererek resmi çiziyor
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(resim_siniflari[y[index]])
    
plot_sample(X_test,y_test,0)
plot_sample(X_test,y_test,1)
plot_sample(X_test,y_test,3)

#%% NORMALIZATION

# Verilerimizi normalize etmemiz gerekiyor. Aksi takdirde CNN algoritmaları yanlıs sonuc verebiliyor. Fotograflar RGB olarak 3 kanal ve her bir pixel 0-255 arasında değer aldıgı için normalization için basitçe her bir pixel degerini 255'e bölmemiz yeterli.

X_train = X_train / 255
X_test = X_test / 255
 
# CNN algoritması normalization istiyor.

#%% DEEP LEARNING ALGORİTMAMIZIN (CNN - Convolutional Neural Network) Tasarımını Yapıyoruz:
    
deep_learning_model = models.Sequential([
    # ilk bölüm convolution layer.. Bu kısımda fotograflardan tanımlama yapabilmek için özellikleri çıkartıyoruz.. (feature extraction)
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),  # input shape'dekiler resmin özellikleri
    layers.MaxPooling2D((2,2)),               # Pooling işlemi overfit işlemi için ve cnn modelinin daha hızlı calısması için kullanılır. (2,2)'ye sıkıştırdık. 
    
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    # İkinci bölüm klasik Artificial Neural Network olan layerlarımız. Yukarıdaki özelliklerimiz ve training bilgilerine göre ANN modelimizi eğiteceğiz
    layers.Flatten(),           # flatten ile de cnn ve ann'yi otomatik olarak baglıyoruz.  tensorflow1 varken manuel yapılıyordu bu işlem, tf2 de otomatikleşti
    layers.Dense(64,activation='relu'),     # hidden layer   # 64 nöron var
    layers.Dense(10,activation='softmax')   # output layer   # 1 output var onda da 10 nöron var, 10 column var diye 
])


deep_learning_model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',    # cıktı direkt rakam değil array oldugu için bu loss fonksiyonunu tercih ettik
                            metrics=['accuracy'])



#%% MODEL FITTING

deep_learning_model.fit(X_train, y_train, epochs=5)     # model fit işlemi    # 5 epochdaki başarısı hesapladık
deep_learning_model.evaluate(X_test,y_test)           # gerçek basarı için testtekilere bakmak lazım # accuracy'i arttırmak için -- epochs sayısı arttırma, filterları değiştirme, nöron sayısını degistirme yapabiliriz.

y_pred = deep_learning_model.predict(X_test)      # predictionu x_test için calıstırıp y predi olusturuyoruz 
y_pred[:3]            # ilk 3 degeri yazmasını istiyoruz

y_predictions_classes = [np.argmax(element) for element in y_pred]      # üstteki 2 boyutlu arrayi daha iyi anlamak için argmax fonksiyonuyla düzelttik
y_predictions_classes[:3] 

y_test[:3]    # burda da 2 üst satırdaki sonucların gerçekteki değerleri gözüküyor. 3 deger ne kadar dogru karsılastırabiliriz

plot_sample(X_test, y_test,0)      # gerçek degeri
resim_siniflari[y_predictions_classes[0]]    # tahmin degeri

plot_sample(X_test,y_test,1)
resim_siniflari[y_predictions_classes[1]]

plot_sample(X_test,y_test,2)
resim_siniflari[y_predictions_classes[2]]


















