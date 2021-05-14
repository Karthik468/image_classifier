import tensorflow.keras
from tensorflow.python.keras.preprocessing import image
import numpy as np
import os
import shutil
import tensorflow as tf
#from PIL import Image
#import matplotlib.pyplot as plt

model = tensorflow.keras.models.load_model('C:/Users/HP/Desktop/traindata notes/modelCNN.h5',compile =False)
model.trainable = False

path = input("Provide path to the Folder of IMAGES:")

personal_img = os.path.join(path + "personal_pics/")
notes_img = os.path.join(path + "notes_pics/")

os.mkdir(personal_img)
os.mkdir(notes_img)
# plt.figure(figsize=(10, 10))
for img in os.listdir(path):
  if img.lower().endswith(('.png', '.jpg', '.jpeg')):
    img_path = os.path.join(path+img)
    try:
      img_t = tf.keras.preprocessing.image.load_img(img_path, grayscale=False, color_mode='rgb', target_size=(160,160,3))
      img_pred = tf.keras.preprocessing.image.img_to_array(img_t)
      # plt.imshow(img_pred/255 ,cmap=plt.cm.binary)
      # plt.xlabel("snnsi")
      # plt.show()
      img_pred = np.expand_dims(img_pred, axis = 0)
      imgs = np.vstack([img_pred])
      res = model.predict(imgs)
      res = tf.nn.sigmoid(res)
      res = tf.where(res < 0.5, 0, 1)
      if res.numpy()[0] == 1:
        new_img_path = os.path.join(personal_img + img)
        shutil.move(img_path, new_img_path)
        print("Moved to Personal-pics folder")
      else:
        new_img_path = os.path.join(notes_img + img)
        shutil.move(img_path, new_img_path)
        print("Moved to Notes-pic folder")
    except Exception as e:
      print(e)

