
import os 
from flask import Flask, render_template, request
from functions.showcase_prediction import predict_image, showcase_prediction
from keras.models import load_model



app = Flask(__name__)

model = load_model('model/my_model_deep_m__40_epochs')

picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder


@app.route('/', methods=["GET"])
def home():
    return render_template('home.html')


@app.route('/', methods=["POST", "GET"])
def predict():
                
    imagefile = request.files['imagefile']
    
    try:
        # used to upload the photo
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(image_path)                                          

        #generate predictions
        predict = predict_image(image_path, model)
        demo_img_path = showcase_prediction(predict, image_path)
    
    except IsADirectoryError:
        demo_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image-not-found.jpeg')
    
    return render_template('home.html', img_path = demo_img_path )


if __name__ =='__main__':
	app.run(port = 4000, debug = True)
