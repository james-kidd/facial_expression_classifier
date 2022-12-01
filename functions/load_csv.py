import os

import pandas as pd
import numpy as np

#load csv
def face_csv_to_df(csv_pathname):
    
    directory = os.getcwd() #returns path to current directory
    try:
        #create dataframe containing image expression label number, usage, and pixels.
        facedata = pd.read_csv(os.path.join(directory, csv_pathname))
        print("Dataframe of CSV successfully created.")
    except:
        print("Issue raised creating dataframe.")
        return 0
    
    facedata = facedata.rename(columns=lambda x: x.strip().lower())
    print(f"initial columns in dataframe: {facedata.columns}")
    
    #create column for emotion label to increase interpretability
    facial_expression = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'} #dictionary to map label names
    facedata['label_name'] = facedata['emotion'].map(facial_expression)
    print("label_name column added to dataframe.")
    
    #relabel public and private test usage rows to solely test
    for usage_type in ['PrivateTest', 'PublicTest']: 
        facedata.loc[(facedata.usage == usage_type),'usage'] = 'Test'
    print("Usage column updated.")
    
    
    #convert string of pixel data to array reshaped.
    img_shape = (48,48)
    facedata['pixels'] = facedata['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48,48))
    print("pixels column converted to numpy array.")
    
    return facedata