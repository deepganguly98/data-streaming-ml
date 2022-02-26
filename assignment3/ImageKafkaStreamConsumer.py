from kafka import KafkaConsumer
import base64
from io import BytesIO
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

def prep_image(msg):
    im_file = BytesIO(base64.b64decode(msg))  # convert image to file-like object
    img = Image.open(im_file).convert('L')
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

def predict_label(img):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    pred = model.predict(img)
    x = np.argmax(pred)
    return class_names[x]

# load model
model = load_model('mnist_fashion_predict.h5')
i=0
my_consumer = KafkaConsumer('predict',bootstrap_servers ='localhost : 9092',auto_offset_reset = 'latest')
for message in my_consumer:  
    message = message.value
    im_d64 = message[1:-1]
    img = prep_image(im_d64)
    label = predict_label(img)
    print(i,"-",type(img)," Predicted Label = ",label)
    i=i+1