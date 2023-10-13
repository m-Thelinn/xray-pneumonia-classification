import matplotlib.pyplot as plt
import numpy as np
import glob
import collections
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Model
from keras.layers import Input, Rescaling, Conv2D, Activation, MaxPooling2D, Flatten, Dense

BATCH_SIZE = 50
NB_CLASSES = 2 #Dos clases: normal(sano) y neumonia
EPOCHS = 35
IMG_ROWS, IMG_COLS = 32, 32 #Escalamos las imagenes a 32x32

datagen = ImageDataGenerator( #Este objeto permite generar lotes de datos aleatorios que se añaden al dataset (rota la imagen, la desplaza, hace zoom...) para mejorar el entrenamiento
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

def load_data():
    name_classes = ['NORMAL', 'PNEUMONIA']

    X, y = [], []
    print("Cargando datos...")
    for class_number, class_name in enumerate(name_classes):
        for filename in glob.glob(f'./dataset/{class_name}/*.jpg'):
            im = image.load_img(filename, target_size=[IMG_ROWS, IMG_COLS], color_mode='grayscale')
            X.append(image.img_to_array(im))
            y.append(class_number)


    input_shape = (IMG_ROWS, IMG_COLS, 1)

    return np.array(X), np.array(y), input_shape

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

def cnn_model(input_shape):
    input_layer = Input(shape=input_shape)

    x = Rescaling(1./255)(input_layer)

    x = Conv2D(6, (5, 5))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (5, 5))(x)
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

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    datagen.fit(X_train)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)

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

    return y_pred_int

#MAIN
X, y, input_shape = load_data()

print(X.shape, 'Total samples')
print(IMG_ROWS,'X', IMG_COLS, 'Image size')
print(input_shape,'Input_shape')
print(EPOCHS,'Epochs')

plot_symbols(X, y)

collections.Counter(y)

print('N samples, Witdh, Height, Channels', X.shape)

# Validacion 10-CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123) #Este objeto permite tener 1 para test y 9 para entrenamiento
accuracies = []
rocs = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = cnn_model(input_shape)
    print(model.summary())
    y_pred_int = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

print(f'Mean Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}')
print(f'Mean ROC AUC: {np.mean(rocs):.4f} +/- {np.std(rocs):.4f}')

print('Metrics')
print(metrics.classification_report(y_test, y_pred_int, target_names=['Healthy', 'Pneumonia']))

print('Confusion matrix')
metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, y_pred_int), display_labels=['Healthy', 'Pneumonia']).plot()

plt.show()