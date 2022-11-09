from flask import Flask, render_template, jsonify
import os
from flask import request

from tensorflow.keras.preprocessing import image
import numpy as np
from keras import applications  
import tensorflow as tf

from datetime import datetime 


project_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'

ALLOWED_EXTENSIONS = {'webp', 'tiff', 'png', 'jpg', 'jpeg'}



model = tf.keras.models.load_model('vgg_16.h5')
vgg16 = applications.VGG16(include_top=False, weights='imagenet') 



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if 'photo' not in request.files:
        response = {"status" : 500,"status_msg": "File is not uploaded","message": ""}
        return jsonify(response)

    file = request.files['photo']
    if file.filename == '':
        response = {"status" : 500,"status_msg": "No image Uploaded","message": ""}
        return jsonify(response)
    
    if file and not allowed_file(file.filename):
        response = {"status" : 500,"status_msg": "File extension is not permitted","message": ""}
        return jsonify(response)


    name = str(datetime.now().microsecond) + str(datetime.now().month) + '-' + str(datetime.now().day) +  '.jpg'
    photo = request.files['photo']
    path = os.path.join(app.config['UPLOAD_FOLDER'],name)
    photo.save(path)



    cells = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

    img = image.load_img(path,target_size = (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis =0)
    x /= 255
    bt_prediction = vgg16.predict(x)  
    # print(bt_prediction)
    preds = model.predict(bt_prediction)


    print(preds[0])
    result = str(cells[np.argmax(preds)])

    os.unlink(path)

    response = {"status" : 200,"status_msg": result}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)    
