from json import loads  
from kafka import KafkaConsumer
import base64
from io import BytesIO
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# load model
model = load_model('/Users/deep98/Desktop/vector-test/assignment1/mnist_fashion_predict.h5')

# decodeit = open('new_file.png', 'wb')
i=0
my_consumer = KafkaConsumer('test-topic',bootstrap_servers ='localhost : 9092',auto_offset_reset = 'latest')
for message in my_consumer:  
    message = message.value
    # print(message)
    # decodeit.write()
    # decodeit.write(message[1:-1])
    im_d64 = message[1:-1]
    # decodeit.write(base64.b64decode(im_d64))
    # print(im_d64)
    im_file = BytesIO(base64.b64decode(im_d64))  # convert image to file-like object
    img = Image.open(im_file).convert('L')
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    pred = model.predict(img)
    x = np.argmax(pred)
    print(i,"-",type(img))
    print("Predicted image = ",class_names[x])
    i=i+1
    
# decodeit.close()