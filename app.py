import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS
from flask import Flask,request
import os
app = Flask(__name__)
CORS(app)
model = load_model("saved_model.h5")
from PIL import Image


@app.route("/",methods=["GET","POST"])
def api():
    if request.method == "GET":
        return "<h1>Send image as a post request</h1>"
    else:
        img = request.files['image'].stream
        im = Image.open(img)
        im = im.save('image.jpeg')
        test_image = image.load_img('image.jpeg',target_size=(64,64)) 
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        if result[0][0]== 1 :
            prediction = 'PNEUMONIA'
        else:
            prediction = 'NORMAL'
    os.remove("image.jpeg")
    return prediction


if __name__ == '__main__':
    app.run(debug=True)