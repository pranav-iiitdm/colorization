import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the colorization model and its dependencies
convNet_architecture_path = 'models/colorization_deploy_v2.prototxt'
preTrained_weights_path = 'models/colorization_release_v2.caffemodel'
clusterCenters_path = 'models/hull_pts.npy'

model = cv2.dnn.readNetFromCaffe(convNet_architecture_path, preTrained_weights_path)
points = np.load(clusterCenters_path)
points = points.transpose().reshape(2, 313, 1, 1)
model.getLayer(model.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
model.getLayer(model.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Function to colorize an image
def colorize_image(image_path):
    img = cv2.imread(image_path)
    bw_img = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(bw_img, cv2.COLOR_BGR2LAB)
    resized_img = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized_img)[0]
    L -= 60
    model.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = model.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    return img, colorized

# Route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            input_image, colorized_image = colorize_image(file_path)
            cv2.imwrite('static/input_image.jpg', input_image)
            cv2.imwrite('static/colorized_image.jpg', colorized_image)
            return render_template('index.html', input_image_path=url_for('static', filename='input_image.jpg'),
                                   colorized_image_path=url_for('static', filename='colorized_image.jpg'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)