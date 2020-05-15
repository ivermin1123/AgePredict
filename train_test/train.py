import os
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.models import load_model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


classes = 101

def getImagePixels(image_path):
    img = image.load_img(image_path, grayscale=False, target_size=((224, 224)))
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x

def load_data():
    df = pd.DataFrame(columns=['pixels', 'age'])

    #path to folder UTKFace
    path_to = '/content/data2/UTKFace/'
    for filename in os.listdir(path_to):
        path = path_to + filename
        age = filename.split('_')[0]
        # if len(df.index) > 12000:
        #   continue
        if int(age) <= 100:
            df = df.append({'pixels': getImagePixels(path), 'age': int(age)}, ignore_index=True)
    print(len(df.index))
    target = df['age'].values
    target_classes = keras.utils.to_categorical(target, classes)

    # Them cac vector anh input vao list features
    features = []

    for i in range(0, df.shape[0]):
        features.append(df['pixels'].values[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)
    features /= 255
    #train, test split
    train_x, test_x, train_y, test_y = train_test_split(features, target_classes, test_size=0.30)
    return train_x, test_x, train_y, test_y


def get_model():
    # Khoi tao model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights('/content/drive/My Drive/AgePredict/weights/vgg_face_weights.h5')

    # Dong bang cac layer ko can train
    for layer in model.layers[:-7]:
        layer.trainable = False

    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)

    age_model.compile(loss='categorical_crossentropy'
                      , optimizer=keras.optimizers.Adam()
                      , metrics=['accuracy']
                      )

    return age_model


# Load du lieu
train_x, test_x, train_y, test_y = load_data()
# Load model
age_model = get_model()
scores = []
# So epoch va batch_size
# epochs = len(total_img_data)/batch_size
epochs = 100
batch_size = 256

#Create checkpoint to save if val_loss decrease
check_point = ModelCheckpoint(
    filepath='/content/drive/My Drive/AgePredict/weights/model_age.hdf5'
    , monitor="val_loss"
    , verbose=1
    , save_best_only=True
    , mode='auto'
)

# train
for i in range(epochs):
    print("Train: ", i)
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
    score = age_model.fit(
        train_x[ix_train], train_y[ix_train]
        , epochs=1
        , validation_data=(test_x, test_y)
        , callback = [check_point]
    )
scores.append(score)


# Luu model
age_model = load_model("/content/gdrive/My Drive/AgePredict/weights/model_age.hdf5")
age_model.save_weights('/content/drive/My Drive/AgePredict/weights/model_age_weights.h5')