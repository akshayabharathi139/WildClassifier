import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report

# Step 1: Set dataset path
dataset_path = 'dataset'

# Step 2: Data Augmentation & Preprocessing
img_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Step 3: Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
epochs = 15
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# ✅ Step 6: Save the model (this is your main save)
os.makedirs('model', exist_ok=True)
model.save('model/animal_classifier.keras')
print("✅ Model saved successfully at model/animal_classifier.keras")

# Step 7: Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Step 8: Predict on new image
def predict_animal(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    class_label = list(train_data.class_indices.keys())[class_idx]
    confidence = np.max(prediction[0]) * 100
    print(f"Predicted Animal: {class_label}")
    print(f"Accuracy: {confidence:.2f}%")

# Example usage
predict_animal(r'C:\Users\Aakan\OneDrive\Desktop\animal1\dataset\Kangaroo\Kangaroo_7_1.jpg')
