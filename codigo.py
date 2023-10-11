import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import collections
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from keras.preprocessing import image
from keras import models
from keras import layers

BATCH_SIZE = 64
NB_CLASSES = 2
EPOCHS = 50

# Scaling input image to theses dimensions
img_rows, img_cols = 32, 32

def load_data():
  name_classes = ['NORMAL','PNEUMONIA']
  X, y  = [], []
  print("Cargando datos...")
  for class_number, class_name in enumerate(name_classes):
    for filename in glob.glob(f'./dataset/{class_name}/*.jpg'):
      im = image.load_img(filename, target_size=[img_rows, img_cols], color_mode = 'grayscale')
      X.append(image.img_to_array(im))
      y.append(class_number)

  input_shape = (img_rows, img_cols, 1)

  return np.array(X), np.array(y), input_shape

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(n, 3))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i],:,:,0])
        plt.gray()
        ax.set_title('{}-{}'.format(y[index[i]],index[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def cnn_model(input_shape):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1./255))

    model.add(layers.Conv2D(6, (5, 5)))
    model.add(layers.Activation("sigmoid"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(16, (5, 5)))
    model.add(layers.Activation("sigmoid"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(120))
    model.add(layers.Activation("sigmoid"))

    model.add(layers.Dense(84))
    model.add(layers.Activation("sigmoid"))

    model.add(layers.Dense(NB_CLASSES))
    model.add(layers.Activation('softmax'))

    return model

#MAIN
X, y, input_shape = load_data()

print(X.shape, 'train samples')
print(img_rows,'x', img_cols, 'image size')
print(input_shape,'input_shape')
print(EPOCHS,'epochs')

plot_symbols(X, y)

collections.Counter(y)

print('N samples, witdh, height, channels',X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

print(f'x_train {X_train.shape} x_test {X_test.shape}')
print(f'y_train {y_train.shape} y_test {y_test.shape}')

model = cnn_model(input_shape)
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2) #, callbacks=[early_stopping])

loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f'loss: {loss:.2f} acc: {acc:.2f}')

print('Predictions')
y_pred = model.predict(X_test) #Extract prediction per sample and class
print(f'AUC {roc_auc_score(y_test, y_pred[:,1], ):.4f}')
y_pred_int = y_pred.argmax(axis=1)
print(collections.Counter(y_pred_int),'\n')

print('Metrics')
print(metrics.classification_report(y_test, y_pred_int, target_names=['Normal','Pneumonia']))

print('Confusion matrix')
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test,y_pred_int), display_labels=['NORMAL','PNEUMONIA']).plot()
plt.show()