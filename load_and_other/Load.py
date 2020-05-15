from load_and_other.func import detectFace, loadAgeModel, findApparentAge, loadGenderModel, loadEmotionModel, loadRaceModel
import cv2
from keras.preprocessing import image
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

face_cascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_default.xml')
def analyzeImage(img_path):
    '''
    :param img_path: path to img
    :return: int: age, str: race, emo, gender
    '''
    # Nếu type của img_path không phải string or không tồn tại thì báo lỗi
    typeIMG = img_path[-3:]
    if typeIMG == 'jpg' or typeIMG == 'png':
        pass
    else:
        return -4
    if type(img_path) != str:
        print("Đường dẫn đưa vào phải là type string !!! Type path bạn đưa vào là: ", type(img_path))
        return -2
    if not os.path.isfile(img_path):
        print("Chắc chắn rằng đường dẫn: ", img_path, " - tồn tại")
        return -3

    img = detectFace(img_path, (224, 224), False)  # Sử dụng một hàm detect có sẵn
    if type(img) != np.ndarray:
        print("Không tìm thấy gương mặt trong ảnh bạn đưa vào.")
        return -1
    else:
        print("Tìm thấy gương mặt trong ảnh bạn đưa vào.")

    # Age
    print("Thực hiện dự đoán: tuổi")
    age_model = loadAgeModel()
    age_predictions = age_model.predict(img)[0, :]
    apparent_age = findApparentAge(age_predictions)

    # Emotion
    print("Thực hiện dự đoán: cảm xúc")
    emotion_model = loadEmotionModel()
    emotion_labels = ['Giận dữ', 'Chán ghét', 'Sợ hãi', 'Vui vẻ', 'Buồn', 'Ngạc nhiên', 'Bình thường']
    img1 = detectFace(img_path, (48, 48), True)
    emotion_predictions = emotion_model.predict(img1)[0, :]
    emo = emotion_labels[np.argmax(emotion_predictions)]

    # Gender
    print("Thực hiện dự đoán: giới tính")
    gender_model = loadGenderModel()
    gender_prediction = gender_model.predict(img)[0, :]
    if np.argmax(gender_prediction) == 0:
        gender = "Nữ"
    elif np.argmax(gender_prediction) == 1:
        gender = "Nam"

    # Race
    print("Thực hiện dự đoán: chủng tộc")
    race_model = loadRaceModel()
    race_predictions = race_model.predict(img)[0, :]
    race_labels = ['Châu á', 'Châu phi', 'Da đen', 'Da trắng', 'Trung đông', 'Mỹ la tinh']
    race = race_labels[np.argmax(race_predictions)]
    # return
    return apparent_age, gender, emo, race


def analyzeCam():
    cap = cv2.VideoCapture(0)  # capture cam

    while (True):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            if w > 130:  # Bo qua cac mat nho

                # Ve hinh chu nhat quanh mat
                cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128), 1)  # draw rectangle to main image

                # Crop mat
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

                try:
                    # Them magin
                    margin = 30
                    margin_x = int((w * margin) / 100);
                    margin_y = int((h * margin) / 100)
                    detected_face = img[int(y - margin_y):int(y + h + margin_y),
                                    int(x - margin_x):int(x + w + margin_x)]
                except:
                    print("detected face has no margin")

                try:
                    age_model = loadAgeModel()
                    output_indexes = np.array([i for i in range(0, 101)])
                    # Dua mat vao mang predict
                    detected_face = cv2.resize(detected_face, (224, 224))

                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    # Hien thi thong tin tuoi
                    age_distributions = age_model.predict(img_pixels)
                    apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis=1))[0]))

                    # Ve khung thong tin
                    info_box_color = (46, 200, 255)
                    triangle_cnt = np.array(
                        [(x + int(w / 2), y), (x + int(w / 2) - 20, y - 20), (x + int(w / 2) + 20, y - 20)])
                    cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                    cv2.rectangle(img, (x + int(w / 2) - 50, y - 20), (x + int(w / 2) + 50, y - 90), info_box_color,
                                  cv2.FILLED)

                    cv2.putText(img, apparent_age, (x + int(w / 2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255),
                                2)


                except Exception as e:
                    print("exception", str(e))

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()


def analyzeImage1(img_path):
    '''
    :param img_path: path to img - str
    :return: age - int
    '''
    typeIMG = img_path[-3:]
    if typeIMG == 'jpg' or typeIMG == 'png':
        pass
    else:
        return -4
    if type(img_path) != str:
        print("Đường dẫn đưa vào phải là type string !!! Type path bạn đưa vào là: ", type(img_path))
        return -2
    if not os.path.isfile(img_path):
        print("Chắc chắn rằng đường dẫn: ", img_path, " - tồn tại")
        return -3

    # detect Face in img
    img = detectFace(img_path, (224, 224), False)
    if type(img) != np.ndarray:
        print("Không tìm thấy gương mặt trong ảnh bạn đưa vào.")
        return -1
    else:
        print("Tìm thấy gương mặt trong ảnh bạn đưa vào.")
    print("Thực hiện dự đoán: tuổi")
    age_model = loadAgeModel()
    age_predictions = age_model.predict(img)[0, :]
    apparent_age = findApparentAge(age_predictions)
    # return
    return apparent_age

