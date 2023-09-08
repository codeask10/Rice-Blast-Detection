from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.preprocessing import image
#from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define a function to preprocess the image
def preprocess_image(img_path):
    # img = load_img(img_path, target_size=(256, 256))
    # img_array = img_to_array(img)
    # img_batch = np.expand_dims(img_array, axis=0)
    # return img_batch
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Preprocess the resized image
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img_resized.astype('float32') / 256.0

    # Reshape the image to match the input shape of the model
    img= np.expand_dims(img, axis=0)
    return img

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction function
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['image']
        # Save the file to disk
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)
        # Preprocess the image
        img = preprocess_image(file_path)
        # Make a prediction using the model
        prediction = model.predict(img)
        # Get the class label
        if prediction < 0.5:
            label = 'healthy'
        else:
            label = 'infected'
        # Return the prediction result to the user
        img1 = cv2.imread(file_path)
        img_resized = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)

        # Preprocess the resized image
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Get the indices of predictions above the threshold
        lesion_indices = np.where(prediction > 0.5)[0]


        # Threshold the grayscale image
        _, binary = cv2.threshold(img_resized, 0, 256, cv2.THRESH_BINARY)

        # Convert the binary image to a single-channel image
        binary_single_channel = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

        # Find connected components in the single-channel image
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_single_channel, connectivity=4)
        # Convert the image to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        save_directory = 'static/uploads/'
        file_name = 'image_rgb.jpg'
        save_path = os.path.join(save_directory, file_name)
        cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        file_path=save_directory+file_name
        return render_template('result.html', label=label, image=file_path)

if __name__ == '__main__':
    app.run(debug=True)
