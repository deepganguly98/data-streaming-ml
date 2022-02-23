# importing the required libraries  
from time import sleep  
from json import dumps  
from kafka import KafkaProducer  
import base64

  
  
with open("assignment2/food.png", "rb") as image2string:
    converted_string = base64.b64encode(image2string.read()).decode('utf-8')
    # converted_string = image2string.read()
    # converted_string = converted_string
# print(converted_string)
  
# with open('encode.bin', "wb") as file:
#     file.write(converted_string)


# initializing the Kafka producer  
my_producer = KafkaProducer(bootstrap_servers = ['localhost:9092'],value_serializer = lambda x:dumps(x).encode('utf-8'))  
# generating the numbers ranging from 1 to 500  
# for n in range(500):  
#     my_data = {'num' : n}  
#     my_producer.send('test-topic', value = my_data)  
#     sleep(1)

# data = {'img' : converted_string}
# print(data)
my_producer.send('test-topic', value = converted_string)
my_producer.flush()

