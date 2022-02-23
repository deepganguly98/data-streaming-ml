# from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
import base64
from PIL import Image
import numpy as np
from io import BytesIO
from json import dumps  
from kafka import KafkaProducer 

(trainX, trainY) , (testX, testY) = fashion_mnist.load_data()

my_producer = KafkaProducer(bootstrap_servers = ['localhost:9092'],value_serializer = lambda x:dumps(x).encode('utf-8'))

for i in range(0,testX.shape[0]):
    im = Image.fromarray(np.uint8(trainX[i])).convert('RGB')
    print(i," ",type(im))
    # im.show()
    im_file = BytesIO()
    im.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    print(im_b64)
    my_producer.send('test-topic', value = im_b64)
    my_producer.flush()
