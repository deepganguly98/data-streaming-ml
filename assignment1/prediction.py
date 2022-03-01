from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
import streamlit as st
import seaborn as sns
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def load_image(filename):
    # load the image
	img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


# load an image and predict the class

# load the image
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# load model
model = load_model('/Users/deep98/Desktop/vector-test/assignment1/mnist_fashion_predict.h5')

# predict the class
tqdm_callback = tfa.callbacks.TQDMProgressBar()



sns.set_theme(style="darkgrid")
sns.set()
st.title('Fashion MNIST Image Classifer using CNN')

def save_uploaded_file(uploaded_file):
    try:
        with open(uploaded_file.name,'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file): 
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # display_image=load_image(display_image)
        img = load_image(uploaded_file.name)
        prediction = model.predict(img,callbacks=[tqdm_callback])

        os.remove(uploaded_file.name)

        # deleting uploaded saved picture after prediction

        # drawing graphs

        st.text('Prediction :')
        answer = class_names[np.argmax(prediction)]
        st.text(answer)

        # fig, ax = plt.subplots()

        # ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)

        # ax.set(xlabel='Confidence %', ylabel='Breed')

        fig = plt.figure(1)
        plot_value_array(1, prediction[0], range(10))
        _ = plt.xticks(range(10), class_names, rotation=45)
        st.pyplot(fig,size=2,aspect=1)