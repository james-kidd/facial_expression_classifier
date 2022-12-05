import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_dataset_share(face_df):
    distribution_labels = face_df.groupby('usage')['label_name'].value_counts()
    N = distribution_labels.groupby(level='usage').count()[0] #number of facial expression classes

    #count of class instances in each usage case (test, train)
    test_data_count = distribution_labels.Test.values
    train_data_count = distribution_labels.Training.values
    
    plt.style.use('ggplot')
    
    ind = np.arange(N) # the x locations for the classes
    width = 0.5
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    #plot bars
    test_bar = ax.bar(ind, test_data_count, width)
    train_bar = ax.bar(ind, train_data_count, width, bottom=test_data_count)
    
    #plot percentage
    height = face_df['label_name'].value_counts()
    percentage = round(height/len(face_df) *100, 2)

    for k, p in enumerate(train_bar):
        ax.text(x=p.get_x() + p.get_width() / 2, y= height[k]+100,
          s="{}%".format(percentage[k]),
          ha='center')
    
    ax.set_title('Image count by emotion and usage')
    ax.set_xticks(ind, (distribution_labels.Test.keys()))
    
    ax.set_ylabel('Image Count')
    tick_no = (distribution_labels.max())
    ax.set_yticks(np.arange(0, tick_no*1.4, tick_no//10))
    
    ax.legend(labels=['Test Data', 'Train Data'])
    
    plt.show()

def showcase_images(face_df):
    plt.figure(figsize=(10,10))
    
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(face_df['pixels'].iloc[i])
        plt.xlabel(face_df['label_name'].iloc[i])
    plt.show()


    
def showcase_augmented_image(dataset, aug_sequence):
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.suptitle("Example of an image augmented", fontsize=14, color = 'white')
    for image, _ in dataset.take(1):
        for i in range(9):
            augmented_images = aug_sequence(image)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

def create_confusion_matrix(test_ds, model, class_names, depth):
    predictions = model.predict(test_ds)
    prediction_class = np.argmax(predictions, axis = 1) 
    
    true_class = []
    for label in test_ds.unbatch():
        true_class.append(label[1].numpy())
        
    mat = confusion_matrix(true_class, prediction_class, labels = np.arange(0,7,1))
    
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.suptitle(f'Confusion Matrix : {depth}', fontsize=12)
    plt.ylabel('true emotion')
    plt.xlabel('predicted emotion')