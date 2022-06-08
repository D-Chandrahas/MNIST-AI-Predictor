import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.fit(train_images, train_labels, epochs=30),"\n\n")

model.save("file_name")

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
