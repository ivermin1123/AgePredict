import numpy as np
from load_and_other.func import loadAgeModel
import matplotlib.pyplot as plt
from keras.preprocessing import image

age_model = loadAgeModel()


def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255
    return test_img


#path to img_test
picture = ""
prediction = age_model.predict(loadImage(picture))


y_pos = np.arange(101)
plt.bar(y_pos, prediction[0], align='center', alpha=0.3)
plt.ylabel('percentage')
plt.title('age')
plt.show()

img = image.load_img(picture)
plt.imshow(img)
plt.show()


output_indexes = np.array([i for i in range(0, 101)])
print("most dominant age class (not apparent age): ", np.argmax(prediction))
apparent_age = np.round(np.sum(prediction * output_indexes, axis=1))
print("apparent age: ", int(apparent_age[0]))