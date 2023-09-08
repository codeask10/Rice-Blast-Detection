<h1>Rice Blast Detection Project</h1>

**Overview**

The Rice Blast Detection Project is an **AI and Machine Learning-based** solution for the early detection of rice blast diseases in rice using computer vision techniques. This project aims to help farmers and researchers identify and manage rice blast infections in their crops more effectively.

**Project Structure**

The project is structured as follows:

Model Training: This section includes code for training the deep learning model used for rice blast detection. It utilizes TensorFlow/Keras and image data augmentation techniques.

**Flask Web Application:**

The web application is built using Flask, a Python web framework. Users can upload rice leaf images, and the application will predict whether the leaf is healthy or infected with rice blasts.

**Getting Started**

To get started with this project, follow these steps:

**Clone the Repository:**

```bash
$ git clone https://github.com/your-username/Rice-Blast-Detection.git
$cd Rice-Blast-Detection
```
**Install Dependencies:**

You may need to install the necessary Python packages. Use pip for this:
```
$pip install -r requirements.txt
```

***Data Preparation:***

Organize your image data into three directories: train, val, and test. Each directory should contain subdirectories for each class (e.g., healthy and infected) with corresponding images.

***Model Training:***

Use the provided Python script to train the model. Customize the hyperparameters and data paths as needed.
```
$ python models.py
```

***Web Application:***

Customize the Flask application as needed.
Ensure that the trained ***model (model.h5)*** is saved in the appropriate directory.

***Run the Web Application:***
```
$python app.py
```

The web application should be accessible at http://localhost:5000  in your web browser.

***Usage**
Visit the web application (http://localhost:5000) and upload a rice leaf image for prediction.
The application will provide a prediction, indicating whether the leaf is healthy or infected with rice blasts.
Additionally, it will display an image highlighting the detected lesions.

<h3>Results and Visualizations</h3>

The project includes visualizations of training and validation accuracy as well as training and validation loss. These can be found in the train_model.py script.

<h3>Acknowledgments</h3>
This project was inspired by the need for early rice blast disease detection in agriculture.
We acknowledge the contributions of the open-source community, including TensorFlow, Keras, Flask, and OpenCV.

<h3>Contact</h3>
Annajmussaquib Khan
Email: annajmussaquib123@gmail.com




