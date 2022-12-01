import pandas as pd
import numpy as np

def select_files_to_delete(duplicates_keys, key_to_img, face_df):
    
    num_of_file_deleted = 0
    files_delete = []
    
    emotion_count = face_df['emotion'].value_counts() #number of images within each emotion class
    zero_hashed_key = hash(np.zeros((48, 48)).tobytes()) #key from hashing corrupted file 
    
    for key in duplicates_keys.index:
        duplicates_idx = key_to_img[(key_to_img == key)].index #get indexed of duplicates
        
        if (zero_hashed_key != key):
            #remove duplicates while keeping unique case
            #based off which category has the smallest dataset size in the emotion classes
            
            emotion_series = face_df['emotion'].iloc[duplicates_idx]
            emotion_to_keep = emotion_count[emotion_series.unique()].idxmin()
            
            #keep 1 image in dataset, therefore remove from to be deleted list
            idx_to_keep = emotion_series.loc[emotion_series==emotion_to_keep].index[0]
            duplicates_idx = duplicates_idx.tolist()
            duplicates_idx.remove(idx_to_keep)
        
        else: 
            #all files will be deleted as they contain no image
            print(f"{len(duplicates_idx)} files were corrupted. Deleting corrupted files ...")

        files_delete.extend(duplicates_idx)
        num_of_file_deleted += len(duplicates_idx)
        
    return files_delete, num_of_file_deleted

#remove image duplicates efficiently
def delete_duplicate_img(face_df): 
    pixel_series = face_df['pixels']
    
    #give image pixels unique idendity (key) through hashing
    key_to_image = pixel_series.apply(lambda x: hash(x.tobytes()))
    
    #list out the image keys with occurances larger than 1
    unique_count = key_to_image.value_counts()
    key_to_duplicates = unique_count.loc[unique_count>1] 
    
    #find which images in duplicates list to delete
    discard_files, num_removed_files = select_files_to_delete(key_to_duplicates, key_to_image, face_df)
        
    face_df.drop(discard_files, axis=0, inplace=True)
    print(f"{num_removed_files} duplicates found. Deleting duplicates...")
    print(f"{len(face_df)} images remaining...")
    
    #add index for later file management
    face_df['index'] = face_df.index
    return face_df
