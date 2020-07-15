import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask
import os
import base64
from flask import request,jsonify
app = Flask(__name__)
import tempfile
model = load_model("saved_model.h5")
import binascii
from PIL import Image



@app.route("/",methods=["POST"])
def home():
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