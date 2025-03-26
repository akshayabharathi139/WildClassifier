import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ✅ Load the trained model
model = load_model('model/animal_classifier.keras')
print("✅ Model loaded successfully!")

# ✅ Set image size (same as training)
img_size = (150, 150)

# ✅ Get class labels from the training directory
dataset_path = 'dataset'
class_labels = sorted(os.listdir(dataset_path))
print(f"✅ Available classes: {class_labels}")

# ✅ Prediction function
def predict_animal(img_path):
    try:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        predicted_class = class_labels[class_idx]

        print(f"🐾 Predicted Animal: {predicted_class}")
        print(f"🎯 Accuracy: {confidence:.2f}%")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

# ✅ Example usage
if __name__ == "__main__":
    test_image_path = r'C:\Users\Aakan\OneDrive\Desktop\animal1\dataset\Kangaroo\Kangaroo_7_1.jpg'
    predict_animal(test_image_path)
