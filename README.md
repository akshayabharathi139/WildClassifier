# WildClassifier
🐾 Animal Image Classification Web Application
📌 Project Description
This project is a Machine Learning-based web application that classifies animal images into one of 15 animal categories. It uses a Convolutional Neural Network (CNN) model trained on a dataset containing images of different animals.

The model predicts the animal type from an uploaded image and displays the animal name along with the prediction confidence (accuracy percentage).

The entire project is wrapped in a Flask web application where users can: ✅ Upload an animal image
✅ Get instant prediction results
✅ View the uploaded image along with the prediction



🚀 Features
Trained CNN model with TensorFlow and Keras

Predicts 15 animal classes (e.g., Cat, Dog, Lion, Tiger, Elephant, etc.)

Flask-based user-friendly web interface

Displays the predicted animal and prediction confidence

Image preview after upload

Easy to extend or improve the model



📂 Folder Structure
animal-classifier/
├── model/                    # Trained CNN model (.keras)
│   └── animal_classifier.keras
├── static/
│   └── uploads/              # Uploaded images saved here
├── templates/
│   ├── index.html            # Upload page
│   └── result.html           # Prediction result page
├── dataset/                  # (Optional) Dataset used for training
├── webapp.py                 # Flask application file
├── requirements.txt          # Required Python libraries
└── README.md                 # Project description



⚙️ Technologies Used
Python

TensorFlow / Keras

Flask (Web Framework)

HTML & CSS (for frontend)

Pillow (Image processing)

NumPy

📥 Installation Instructions
git clone https://github.com/yourusername/animal-classifier-webapp.git
cd animal-classifier-webapp
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python webapp.py


http://127.0.0.1:5000/
