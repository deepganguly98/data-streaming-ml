# importing the required modules  
from json import loads  
from kafka import KafkaConsumer
import base64
  
# def forgiving_json_deserializer(v):
#     if v is None:
#     	return
#     try:
#         return loads(v.decode('utf-8'))
#     except json.decoder.JSONDecodeError:
#         log.exception('Unable to decode: %s', v)
#         return None

decodeit = open('new_file.png', 'wb')
# generating the Kafka Consumer  
my_consumer = KafkaConsumer('test-topic',bootstrap_servers ='localhost : 9092',auto_offset_reset = 'latest')
	# ,value_deserializer=forgiving_json_deserializer)

for message in my_consumer:  
	message = message.value
	decodeit.write(base64.b64decode(message[1:-1]))
	# decodeit.write(message[1:-1])
	print(message[1:-1]) 
  


decodeit.close()