import os
import base64
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError

credentials_path = '/Users/deep98/Downloads/vector_test_private_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

timeout = 5.0

subscriber = pubsub_v1.SubscriberClient()
subscription_path = 'projects/heroic-arbor-342611/subscriptions/predict-sub'

img = open("new_file_pubsub.png","wb")

def callback(message):
	print(f'Received message.')
	# print(f'data : {message.data}')
	img_str = message.data
	img.write(base64.b64decode(img_str))
	img.close()
	message.ack()

streaming_pull_future = subscriber.subscribe(subscription_path,callback = callback)
print(f'Listening for messages on {subscription_path}')

with subscriber:
	try:
		streaming_pull_future.result(timeout=timeout)
		# streaming_pull_future.result()
	except TimeoutError:
		streaming_pull_future.cancel()
		streaming_pull_future.result