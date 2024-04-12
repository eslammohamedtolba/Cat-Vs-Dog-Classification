# Load dependencies
from flask import Flask, render_template, request
import numpy as np
import base64
import cv2 # to resize input images
from keras.models import load_model

img_size = 100


# Create application
app = Flask(__name__, template_folder="templates", static_folder="static",static_url_path="/")
# Load model
# model = pkl.load(open('./Preparing_model/model.pkl','rb'))

# Load model
model = load_model('./Preparing_model/my_cat_dog_classifier.h5')


# Create routes and urls
@app.route('/', methods=['GET', 'POST'])
def Home():
    return render_template("index.html", prediction_result = -1)

# Create prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    # Read the image data from FileStorage and convert it to numpy array
    image_data = np.fromstring(image.read(), np.uint8)
    # Decode the image array
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    # Resize and preprocess the image
    resized_image = cv2.resize(img, (img_size, img_size))
    resized_image = resized_image.reshape((1, img_size, img_size, 3))  # Add batch dimension
    resized_image_scaled = resized_image / 255.0  # Normalize pixel values
    # Make prediction
    prediction = model.predict(resized_image_scaled)
    # Get the predicted class (assuming binary classification)
    prediction_result = 'Cat' if prediction[0][0] < 0.5 else 'Dog'
    # Encode the image before passing it into the HTML file
    encoded_image = base64.b64encode(cv2.imencode('.jpg', resized_image[0])[1]).decode('utf-8')
    return render_template('index.html', prediction_result = prediction_result, image=encoded_image)


if __name__ == "__main__":
    app.run(debug=True)





