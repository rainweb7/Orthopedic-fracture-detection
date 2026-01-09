import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    color_mode="grayscale",
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224, 224),
    color_mode="grayscale",
    class_mode="binary",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save(os.path.join(BASE_DIR, "fracture_model.h5"))

# Accuracy graph
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.clf()

# Loss graph
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
