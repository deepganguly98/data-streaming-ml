# Data Streaming and continous prediciton
This reporisitory primarily focuses on establishing data pipelines to consume deep learning service.

## Assignment 1
Implenting a CNN Multiclass Image Classifer for MNIST Fasion dataset and deploying model as a Web Application using streamlit.
Application can be executed by navigating to 'assignment1' directory and running the below command in the terminal :

```streamlit run assignment1/prediction.py```

<img width="565" alt="image" src="https://user-images.githubusercontent.com/33273794/156141451-feae604c-5943-45f7-aed0-1315ec8cdb67.png">


## Assignment 2
Using python to interface Apache Kafka or Google Pub/Sub to stream messages in this case an image is put on the queue by a producer in base 64 format which is consumed by a consumer to reconstruct the image and dump as a file.


## Assignment 3
Using the interfaces created in assignment 2 and model generated in assignment 1 to stream images from the test set of MNIST Fashion dataset to perform continous predictions as the streams are handled by the broker.
The user has the choice to choose either Apache Kafka or Google Pub/Sub as the message broker.
The Producer and Consumer methods are handled by seperate threads thus enabling multithreading for a faster production and delivery.



```
python assignment3/StreamPredict.py --help
usage: StreamPredict.py [-h] choice

Choose message broker

positional arguments:
  choice      1 - choose Apacha Kafka as message broker 2 - choose Google Pub/Sub as message broker

optional arguments:
  -h, --help  show this help message and exit
  ```
  <img width="839" alt="image" src="https://user-images.githubusercontent.com/33273794/156142811-f74342c5-2766-4a23-94c8-3d6546c069e7.png">
<img width="852" alt="image" src="https://user-images.githubusercontent.com/33273794/156142896-256aeb8c-4121-4972-99bf-3e409e838c10.png">
