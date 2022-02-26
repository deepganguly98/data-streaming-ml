from keras.datasets import fashion_mnist
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from json import dumps  
from kafka import KafkaProducer, KafkaConsumer
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import os
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(trainX, trainY) , (testX, testY) = fashion_mnist.load_data()
model = load_model('/Users/deep98/Desktop/vector-test/assignment3/mnist_fashion_predict.h5')
my_producer = KafkaProducer(bootstrap_servers = ['localhost:9092'],value_serializer = lambda x:dumps(x).encode('utf-8'))
my_consumer = KafkaConsumer('predict',bootstrap_servers ='localhost : 9092',auto_offset_reset = 'latest')


def img_conv(imgarr):
    im = Image.fromarray(np.uint8(imgarr)).convert('RGB')
    im_file = BytesIO()
    im.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64


def produce():
    print("Publishing images...")
    for i in tqdm(range(0,250)): 
        im_b64 = img_conv(testX[i])
        # print(im_b64)
        my_producer.send('predict', value = im_b64)
        my_producer.flush()

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

def consume():
    print("Prediction about to begin...")
    i=0
    for message in my_consumer:  
        message = message.value
        im_d64 = message[1:-1]
        img = prep_image(im_d64)
        label = predict_label(img)
        print(i,"-",type(img)," Predicted Label = ",label)
        i=i+1

if __name__ == '__main__':
    t_producer = threading.Thread(target=produce)
    t_consumer = threading.Thread(target=consume)
    t_producer.setDaemon(True)
    t_consumer.setDaemon(True)
    t_producer.start()
    t_consumer.start()
    while True:
        pass
