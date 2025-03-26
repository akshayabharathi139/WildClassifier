import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image  # ✅ Force Pillow import
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ✅ Load your trained model
model = load_model('model/animal_classifier.keras')

# ✅ Class labels (you can change based on your dataset)
class_labels = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Donkey', 'Elephant',
                'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # ✅ Ensure the uploads folder exists (This prevents the FileNotFoundError)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # ✅ Save the uploaded file
        file.save(file_path)

        # ✅ Process the image for prediction
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Make prediction
        prediction = model.predict(img_array)[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100
        predicted_class = class_labels[class_idx]

        # ✅ Render the result with image preview
        return render_template('result.html',
                               animal=predicted_class,
                               confidence=f"{confidence:.2f}%",
                               img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)


    
