import os
from google.cloud import pubsub_v1
import base64

credentials_path = '/Users/deep98/Downloads/vector_test_private_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

publisher = pubsub_v1.PublisherClient()
topic_path = 'projects/heroic-arbor-342611/topics/predict'

with open("/Users/deep98/Desktop/vector-test/assignment2/food.png","rb") as img:
	img_str = base64.b64encode(img.read())
# print(img_str)

data = img_str

future = publisher.publish(topic_path,data)
print(f'published message id {future.result()}')
