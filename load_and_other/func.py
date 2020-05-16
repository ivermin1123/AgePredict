import pandas as pd
from keras.preprocessing import image
import cv2
import math
from PIL import Image
import os
from pathlib import Path
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, ZeroPadding2D
import zipfile
import base64


def get_opencv_path():
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    face_detector_path = path + "/data/haarcascade_frontalface_default.xml"
    eye_detector_path = path + "/data/haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")
    return path + "/data/"


def findThreshold(distance_metric):
    threshold = 0.40

    if distance_metric == 'cosine':
        threshold = 0.40
    elif distance_metric == 'euclidean':
        threshold = 0.55
    elif distance_metric == 'euclidean_l2':
        threshold = 0.75

    return threshold

def distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]

    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


def detectFace(img, target_size=(224, 224), grayscale=False, enforce_detection=True):
    # -----------------------

    exact_image = False
    if type(img).__module__ == np.__name__:
        exact_image = True

    base64_img = False
    if len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    # -----------------------

    opencv_path = get_opencv_path()
    face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
    eye_detector_path = opencv_path + "haarcascade_eye.xml"

    if os.path.isfile(face_detector_path) != True:
        raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path,
                         " violated.")

    # --------------------------------

    face_detector = cv2.CascadeClassifier(face_detector_path)
    eye_detector = cv2.CascadeClassifier(eye_detector_path)

    if base64_img == True:
        img = loadBase64Img(img)

    elif exact_image != True:  # image path passed as input

        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    img_raw = img.copy()

    # --------------------------------

    faces = face_detector.detectMultiScale(img, 1.3, 5)

    # print("found faces in ",image_path," is ",len(faces))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]
        detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        # ---------------------------
        # face alignment

        eyes = eye_detector.detectMultiScale(detected_face_gray)

        if len(eyes) >= 2:
            # find the largest 2 eye
            base_eyes = eyes[:, 2]

            items = []
            for i in range(0, len(base_eyes)):
                item = (base_eyes[i], i)
                items.append(item)

            df = pd.DataFrame(items, columns=["length", "idx"]).sort_values(by=['length'], ascending=False)

            eyes = eyes[df.idx.values[0:2]]

            # -----------------------
            # decide left and right eye

            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # -----------------------
            # find center of eyes

            left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]

            right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]

            # -----------------------
            # find rotation direction

            if left_eye_y > right_eye_y:
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate same direction to clock
            else:
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock

            # -----------------------
            # find length of triangle edges

            a = distance(left_eye_center, point_3rd)
            b = distance(right_eye_center, point_3rd)
            c = distance(right_eye_center, left_eye_center)

            # -----------------------
            # apply cosine rule

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # -----------------------
            # rotate base image

            if direction == -1:
                angle = 90 - angle

            img = Image.fromarray(img_raw)
            img = np.array(img.rotate(direction * angle))

            # you recover the base image and face detection disappeared. apply again.
            faces = face_detector.detectMultiScale(img, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # -----------------------

        # face alignment block end
        # ---------------------------

        # face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        detected_face = cv2.resize(detected_face, target_size)

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # normalize input in [0, 1]
        img_pixels /= 255

        return img_pixels

    else:

        if (exact_image == True) or (enforce_detection != True):

            if grayscale == True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, target_size)
            img_pixels = image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            return img_pixels
        else:
            raise ValueError("Face could not be detected. Please confirm that the picture is a face photo.")


def loadRaceModel():
    model = loadVggFaceModel()

    # --------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    if os.path.isfile('./weights/race_model_single_batch.h5') != True:
        print("race_model_single_batch.h5 will be downloaded...")
        # zip
        url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
        output = './weights/race_model_single_batch.zip'
        gdown.download(url, output, quiet=False)

        # unzip race_model_single_batch.zip
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('./weights/')

    race_model.load_weights('./weights/race_model_single_batch.h5')

    return race_model


def loadEmotionModel():
    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    # ----------------------------

    if os.path.isfile('./weights/facial_expression_model_weights.h5') != True:
        print("facial_expression_model_weights.h5 will be downloaded...")

        # TO-DO: upload weights to google drive

        # zip
        url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
        output = './weights/facial_expression_model_weights.zip'
        gdown.download(url, output, quiet=False)

        # unzip facial_expression_model_weights.zip
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('./weights/')

    model.load_weights('./weights/facial_expression_model_weights.h5')

    return model


def loadGenderModel():
    model = loadVggFaceModel()

    # --------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    home = str(Path.home())

    if os.path.isfile('./weights/gender_model_weights.h5') != True:
        print("gender_model_weights.h5 will be downloaded...")

        url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
        output = './weights/gender_model_weights.h5'
        gdown.download(url, output, quiet=False)

    gender_model.load_weights('./weights/gender_model_weights.h5')

    return gender_model


def loadAgeModel():
    model = loadVggFaceModel()

    # --------------------------
    #Age from 1-100
    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights
    if os.path.isfile('./weights/model_age_weights.h5') != True:
        print("age_model_weights.h5 will be downloaded...")
        url = 'https://drive.google.com/uc?id=102b8Jl2S-5Hc0fD4fHo0RL8sR8tUrJEh'
        output = './weights/age_model_weights.h5'
        gdown.download(url, output, quiet=False)

    age_model.load_weights('./weights/model_age_weights.h5')

    return age_model


def loadVerify():
    model = loadVggFaceModel()

    # -----------------------------------

    if os.path.isfile('./weights/vgg_face_weights.h5') != True:
        print("vgg_face_weights.h5 will be downloaded...")

        url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'
        output = './weights/vgg_face_weights.h5'
        gdown.download(url, output, quiet=False)

    # -----------------------------------

    model.load_weights('./weights/vgg_face_weights.h5')

    # -----------------------------------

    # TO-DO: why?
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor
# --------------------------

def findApparentAge(age_predictions):
    output_indexes = np.array([i for i in range(0, 101)])
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age

def loadVggFaceModel():
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

    return model

