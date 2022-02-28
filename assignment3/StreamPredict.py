from keras.datasets import fashion_mnist
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from json import dumps  
from kafka import KafkaProducer, KafkaConsumer
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import os
import argparse
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def img_conv(imgarr):
    im = Image.fromarray(np.uint8(imgarr)).convert('RGB')
    im_file = BytesIO()
    im.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64

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

def pubsub_produce():
    print("Publishing images...")
    for i in tqdm(range(0,50)): 
        im_b64 = img_conv(testX[i]).encode('utf-8')
        # print(im_b64)
        future = publisher.publish(topic_path,im_b64)
        print(f'published message id {future.result()}')

def kafka_produce():
    print("Publishing images...")
    for i in tqdm(range(0,50)): 
        im_b64 = img_conv(testX[i])
        # print(im_b64)
        my_producer.send('predict', value = im_b64)
        my_producer.flush()

def kafka_consume():
    print("Prediction about to begin...")
    i=0
    for message in my_consumer:  
        message = message.value
        im_d64 = message[1:-1]
        img = prep_image(im_d64)
        label = predict_label(img)
        print(i,"-",type(img)," Predicted Label = ",label)
        i=i+1
    KafkaConsumer.close()

def pubsub_callback(message):
    # print(f'Received message.')
    # print(f'data : {message.data}')
    img_str = message.data
    # img.write(base64.b64decode(img_str))
    img = prep_image(img_str)
    label = predict_label(img)
    print(type(img)," Predicted Label = ",label)
    # print(img_str)
    # img.close()
    message.ack()

def pubsub_consume():
    streaming_pull_future = subscriber.subscribe(subscription_path,callback = pubsub_callback)
    print(f'Listening for messages on {subscription_path}')

    with subscriber:
        try:
            # streaming_pull_future.result(timeout=timeout)
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()
            streaming_pull_future.result

def produce(choice):
    if choice == 1:
        kafka_produce()
    elif choice == 2:
        pubsub_produce()

def consume(choice):
    if choice == 1:
        kafka_consume()
    elif choice == 2:
        pubsub_consume()

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(description='Choose message broker')
    my_parser.add_argument('choice',type=int, help="1 - choose Apacha Kafka as message broker \n2 - choose Google Pub/Sub as message broker")
    args = my_parser.parse_args()
    choice = args.choice

    if choice == 1:
        print("Broker - Apache Kafka")
        os.system("brew services start kafka")
        os.system("brew services start zookeeper")
        os.system("kafka-topics --create --topic predict --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1")
        # os.system("kafka-topics --delete --topic predict --bootstrap-server localhost:9092")
        my_producer = KafkaProducer(bootstrap_servers = ['localhost:9092'],value_serializer = lambda x:dumps(x).encode('utf-8'))
        my_consumer = KafkaConsumer('predict',bootstrap_servers ='localhost : 9092',auto_offset_reset = 'latest')

    elif choice == 2:
        print("Broker - Google Pub/Sub")
        credentials_path = 'vector_test_private_key.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        publisher = pubsub_v1.PublisherClient()
        topic_path = 'projects/heroic-arbor-342611/topics/predict'
        timeout = 5.0
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = 'projects/heroic-arbor-342611/subscriptions/predict-sub'


    model = load_model('assignment1/mnist_fashion_predict.h5')
    (trainX, trainY) , (testX, testY) = fashion_mnist.load_data()
    print("model loaded and brokers configured successfully.")

    # print(os.getcwd())
    # os.system("brew services start kafka")
    # os.system("brew services start zookeeper")
    # os.system("kafka-topics --create --topic predict --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1")
    # os.system("kafka-topics --delete --topic predict --bootstrap-server localhost:9092")
    # os.system("brew services stop kafka")
    # os.system("brew services stop zookeeper")

    t_producer = threading.Thread(target=produce, args = (choice,))
    t_consumer = threading.Thread(target=consume, args = (choice,))
    t_producer.setDaemon(True)
    t_consumer.setDaemon(True)
    t_producer.start()
    t_consumer.start()


    while True:
        pass
    # t_producer.join()
    # t_consumer.join()
