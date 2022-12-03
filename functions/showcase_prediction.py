import tensorflow as tf
import keras

import os

from keras.utils import load_img, img_to_array
# from keras.optimizers import Adam

from keras import layers

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_shape = (48,48) #shape of image
class_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_image(image_path, model):

    img = load_img(
        image_path, target_size=img_shape, color_mode = 'grayscale'
        )

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return predictions

def showcase_prediction(predictions_array, img_path):
    score_value = tf.nn.softmax(predictions_array[0])
    max_index = np.argmax(score_value)
    true_label = class_label[max_index]
    
    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(wspace=0.1)
    plt.subplots_adjust(bottom = 0.18)
    
    plt.suptitle(f"Predicted Emotion: {true_label} ({np.round(np.max(score_value)*100,2)}%)", fontsize=14, color = 'black')
    
    #plot predicting certainty
    ax1 = plt.subplot(1,2,1)
    showcase_prediction = ax1.bar(class_label, (score_value*100), color="#A7C7E7")
    showcase_prediction[max_index].set_color('#1F51FF')
    plt.xticks(rotation = 45)
    plt.ylabel("Certainty (%)")
    
    #plot image
    ax2 = plt.subplot(1,2,2)
    img = load_img(img_path, target_size=img_shape, color_mode = 'grayscale')
    img_array = img_to_array(img)
    ax2.imshow(img_array)
    ax2.axis('off')
    
    new_demo_path = os.path.join(os.path.dirname(img_path),'demo_prediction_'+os.path.basename(img_path))
        
    plt.savefig( new_demo_path )
    
    return new_demo_path
