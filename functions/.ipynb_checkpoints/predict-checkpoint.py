import tensorflow as tf
import keras

def predict_image(image_path, model):

    img = keras.utils.load_img(
        image_path, target_size=img_shape, color_mode = 'grayscale'
        )

    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )