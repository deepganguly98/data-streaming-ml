import base64
from io import BytesIO
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError

model = load_model('/Users/deep98/Desktop/vector-test/assignment3/mnist_fashion_predict.h5')
credentials_path = '/Users/deep98/Downloads/vector_test_private_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

timeout = 5.0

subscriber = pubsub_v1.SubscriberClient()
subscription_path = 'projects/heroic-arbor-342611/subscriptions/predict-sub'
def callback(message):
    i=0
    # print(f'Received message.')
    # print(f'data : {message.data}')
    img_str = message.data
    # img.write(base64.b64decode(img_str))
    img = prep_image(img_str)
    label = predict_label(img)
    print(i,"-",type(img)," Predicted Label = ",label)
    # print(img_str)
    # img.close()
    i=i+1
    message.ack()

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

streaming_pull_future = subscriber.subscribe(subscription_path,callback = callback)
print(f'Listening for messages on {subscription_path}')

with subscriber:
	try:
		streaming_pull_future.result(timeout=timeout)
		# streaming_pull_future.result()
	except TimeoutError:
		streaming_pull_future.cancel()
		streaming_pull_future.result


