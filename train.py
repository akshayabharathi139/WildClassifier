import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    r'C:\Users\Aakan\OneDrive\Desktop\animal1\dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

# Build the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(15, activation='softmax')  # 15 animal classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10)

# ✅ Save the trained model (PUT THIS AFTER model.fit())
os.makedirs('model', exist_ok=True)
model.save('model/animal_classifier.keras')  # ✅ Correct model save

print("Model saved successfully!")
