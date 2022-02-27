# from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import os
from google.cloud import pubsub_v1

credentials_path = '/Users/deep98/Downloads/vector_test_private_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

publisher = pubsub_v1.PublisherClient()
topic_path = 'projects/heroic-arbor-342611/topics/predict'

(trainX, trainY) , (testX, testY) = fashion_mnist.load_data()

def img_conv(imgarr):
    im = Image.fromarray(np.uint8(imgarr)).convert('RGB')
    im_file = BytesIO()
    im.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    return im_b64


# my_producer = KafkaProducer(bootstrap_servers = ['localhost:9092'],value_serializer = lambda x:dumps(x).encode('utf-8'))

for i in range(0,5):
    im_b64 = img_conv(trainX[i])
    # print(im_b64)
    future = publisher.publish(topic_path,im_b64)
    print(f'published message id {future.result()}')

