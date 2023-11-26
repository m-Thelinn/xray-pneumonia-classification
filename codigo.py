'''Miguel Rodríguez Sánchez
Este código esta destinado a la practica de vision artificial de Desafíos de Programación.

Tenemos tres modelos creados
 - LeNet5
 - Arquitectura propia
 - ResNet con fine-tunning

Esta enfocado a CLASIFICACION MULTICLASE, es decir, tenemos 3 posibles clases (sano, viral y bacteriana)
En caso dde querer clasificación binaria, se peuden descomentar las funciones que permiten hacer la clasificacion
binaria.
'''

import matplotlib.pyplot as plt
import numpy as np
import glob
import collections
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.ndimage import median_filter

import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Rescaling, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.applications import ResNet50

tf.compat.v1.disable_v2_behavior()

BATCH_SIZE = 35
EPOCHS = 35
IMG_ROWS, IMG_COLS = 32, 32 #se puede probar con 64x64 o el tamaño de imagen que se desee
NB_CLASSES = 3

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
)

def load_data():
    name_classes = ['NORMAL', 'PNEUMONIA']

    X, y = [], []
    for class_number, class_name in enumerate(name_classes):
        for filename in glob.glob(f'./chest_xray_512/{class_name}/*.jpg'):
            im = image.load_img(filename, target_size=[IMG_ROWS, IMG_COLS], color_mode='grayscale')
            img_x = image.img_to_array(im)

            #Suavizado por filtro de mediana
            img_x = median_filter(img_x, size=2)

            #Aumentamos el contraste
            mean_intensity = np.mean(img_x)
            adjusted_image = (img_x - mean_intensity) * 1.3 + mean_intensity

            X.append(adjusted_image)

            if 'virus' in filename.lower():
                y.append(1)
            elif 'bacteria' in filename.lower():
                y.append(2)
            else:
                y.append(0)

    input_shape = (IMG_ROWS, IMG_COLS, 1)

    return np.array(X), np.array(y), input_shape

'''
def load_data_binary():
  name_classes = ['NORMAL','PNEUMONIA']
  X,y  = [], []
  for class_number, class_name in enumerate(name_classes):
    for filename in glob.glob(f'./chest_xray_512/{class_name}/*.jpg'):
      im = image.load_img(filename, target_size=[IMG_ROWS, IMG_COLS], color_mode = 'grayscale')
      img_x = image.img_to_array(im)

      #Suavizado por filtro de mediana
      img_x = median_filter(img_x, size=2)

      #Aumentamos el contraste
      mean_intensity = np.mean(img_x)
      adjusted_image = (img_x - mean_intensity) * 1.3 + mean_intensity

      X.append(adjusted_image)
      y.append(class_number)

  input_shape = (IMG_ROWS, IMG_COLS, 1)

  return np.array(X), np.array(y), input_shape
'''

def plot_symbols(X, y, n=15):
    index = np.random.randint(len(y), size = n)
    plt.figure(figsize = (n, 3))

    for i in np.arange(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[index[i],:,:,0])
        plt.gray()
        ax.set_title('{} - {}'.format(y[index[i]], index[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def cnn_lenet(input_shape):
    inputs = Input(shape=input_shape)
    x = Rescaling(1./255)(inputs)

    x = Conv2D(6, (5, 5))(x)
    x = Activation("sigmoid")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (5, 5))(x)
    x = Activation("sigmoid")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(120)(x)
    x = Activation("sigmoid")(x)

    x = Dense(84)(x)
    x = Activation("sigmoid")(x)

    outputs = Dense(NB_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def cnn_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Rescaling(1./255)(input_layer)

    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (5, 5))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(8, (5, 5))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(120)(x)
    x = Activation("relu")(x)

    x = Dense(84)(x)
    x = Activation("relu")(x)

    output_layer = Dense(NB_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def cnn_resnet(input_shape):
    resnet = ResNet50(weights='imagenet', include_top=False)

    input_tensor = Input(shape=input_shape)
    x = Conv2D(3, (3,3), padding='same')(input_tensor)
    x = Rescaling(1./255)(x)

    x = resnet(x)

    x = Flatten()(x)

    x = Dense(120)(x)
    x = Activation("relu")(x)

    x = Dense(84)(x)
    x = Activation("relu")(x)

    output_layer = Dense(NB_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_tensor,outputs=output_layer)

    return model

def show_evolution(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(train_acc, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.title('Training')

    plt.subplot(1, 2, 2)

    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy and Loss')
    plt.title('Validation')

    plt.show()

def show_shap_figure():
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    de = shap.DeepExplainer(model, background)
    shap_values = de.shap_values(X_test[1:5])
    shap.image_plot(shap_values, X_test[1:5], width=15)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, rocs):
    datagen.fit(X_train)
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy']) #se puede sustituir por optimizer=Adam(learning_rate=0.00085) para configurar los ajustes

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2, callbacks=[early_stopping])

    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f'Loss: {loss:.2f} Acc: {acc:.2f}')
    accuracies.append(acc)

    print('Predictions')
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred, multi_class='ovr', labels=[0, 1, 2])
    rocs.append(roc)
    print(f'ROC {roc:.4f}')

    y_pred_int = y_pred.argmax(axis=1)
    print(collections.Counter(y_pred_int),'\n')

    return y_test, y_pred_int

'''
def train_and_evaluate_model_binary(model, X_train, y_train, X_test, y_test, rocs):
    datagen.fit(X_train)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2, callbacks=[early_stopping])

    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f'Loss: {loss:.2f} Acc: {acc:.2f}')
    accuracies.append(acc)

    print('Predictions')
    y_pred = model.predict(X_test)
    roc = roc_auc_score(y_test, y_pred[:,1], )
    rocs.append(roc)
    print(f'ROC {roc:.4f}')

    y_pred_int = y_pred.argmax(axis=1)
    print(collections.Counter(y_pred_int),'\n')

    return y_test, y_pred_int
'''

X, y, input_shape = load_data()

print(X.shape, 'Total samples')
print(IMG_ROWS,'X', IMG_COLS, 'Image size')
print(input_shape,'Input_shape')
print(EPOCHS,'Epochs')

plot_symbols(X, y)

collections.Counter(y)

print('N samples, Witdh, Height, Channels', X.shape)

accuracies = []
rocs = []
rocs2 = []
all_y_true = []
all_y_pred = []
all_y_true2 = []
all_y_pred2 = []
wilcoxon_results = []

# Validacion 10-CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

for train_index, test_index in skf.split(X, y):
    model = cnn_lenet(input_shape)
    model2 = cnn_model(input_shape)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(model.summary())
    y_true, y_pred_int = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, rocs)
    #show_shap_figure()
    print(model2.summary())
    y_true2, y_pred_int2 = train_and_evaluate_model(model2, X_train, y_train, X_test, y_test, rocs2)

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred_int)
    all_y_true2.extend(y_true2)
    all_y_pred2.extend(y_pred_int2)

print(rocs)
print(rocs2)
model1 = 'Arquitectura ResNet50'
model2 = 'Arquitectura propia'

plt.plot(range(len(rocs)),rocs,'ro-',label=model1)
plt.plot(range(len(rocs2)),rocs2,'bs-',label=model2)
plt.ylabel('ROC')
plt.xlabel('Fold')
plt.legend()

wilcox_W, p_value = wilcoxon(rocs, rocs2, alternative='greater', zero_method='wilcox', correction=False)
print(f"El valor de la estadística de prueba (Wilcox W) es {wilcox_W:.2f}, y el p-valor es {p_value:.2f}.")
if p_value < 0.05:
    print("Hay evidencia estadística para rechazar la hipótesis nula. Hay diferencias significativas entre los dos modelos.")
else:
    print("No hay evidencia estadística para rechazar la hipótesis nula. No hay diferencias significativas entre los dos modelos.")

statistic, p_value = mannwhitneyu(rocs2, rocs)
print(f"El valor de la estadística de prueba Mann-Whitney es {statistic:.2f}, y el p-valor es {p_value:.2f}.")
if p_value < 0.05:
    print("Hay evidencia estadística para rechazar la hipótesis nula. Hay diferencias significativas entre los dos modelos.")
else:
    print("No hay evidencia estadística para rechazar la hipótesis nula. No hay diferencias significativas entre los dos modelos.")

print(f'Metrics {model1}')
print(f'Mean ROC AUC: {np.mean(rocs):.4f} +/- {np.std(rocs):.4f}')
print('Metrics')
print(metrics.classification_report(all_y_true, all_y_pred, target_names=['Healthy', 'Pneumo Viral', 'Pneumo Bacteriana']))
conf_matrix = metrics.confusion_matrix(all_y_true, all_y_pred)
metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=['Healthy', 'Pneumo Viral', 'Pneumo Bacteriana']).plot()
#print(metrics.classification_report(all_y_true, all_y_pred, target_names=['Healthy', 'Pneumonia']))
#metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(all_y_true, all_y_pred), display_labels=['Healthy', 'Pneumonia']).plot()

print(f'Metrics {model2}')
print(f'Mean ROC AUC: {np.mean(rocs2):.4f} +/- {np.std(rocs2):.4f}')
print('Metrics')
print(metrics.classification_report(all_y_true2, all_y_pred2, target_names=['Healthy', 'Pneumo Viral', 'Pneumo Bacteriana']))
conf_matrix2 = metrics.confusion_matrix(all_y_true2, all_y_pred2)
metrics.ConfusionMatrixDisplay(conf_matrix2, display_labels=['Healthy', 'Pneumo Viral', 'Pneumo Bacteriana']).plot()
#print(metrics.classification_report(all_y_true2, all_y_pred2, target_names=['Healthy', 'Pneumonia']))
#metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(all_y_true2, all_y_pred2), display_labels=['Healthy', 'Pneumonia']).plot()
plt.show()
show_shap_figure()