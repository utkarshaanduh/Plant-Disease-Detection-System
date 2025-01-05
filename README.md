# Overview

The Plant Disease Detection System is a machine learning-based solution designed to classify plant diseases accurately using image data. This project utilizes a deep convolutional neural network (CNN) to detect 38 different types of plant diseases, aiding farmers and agricultural experts in early detection and treatment.

# Dataset

We used the New Plant Diseases Dataset from Kaggle. The dataset contains labeled images of healthy and diseased plant leaves, which serve as training and validation data for the model.

# Features

Robust Model Architecture: Built using TensorFlow's Sequential API, the CNN comprises multiple convolutional layers with ReLU activations, max-pooling layers, dropout regularization, and dense layers for classification.

### High Accuracy: The model achieved excellent performance metrics:

### Training Accuracy: 98.65%

### Training Loss: 0.0413

### Validation Accuracy: 95.67%

### Validation Loss: 0.1539

### Fine-Tuned Hyperparameters: Optimized using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy loss for multi-class classification.

# Model Architecture
```
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=38, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### How It Works

*Input Preprocessing*: Images are resized to 128x128 and normalized for consistency.

*Feature Extraction*: Convolutional layers extract spatial features from the input images.

*Classification*: Fully connected layers and softmax activation classify the input into one of the 38 disease categories.

# Results

### Metric Value

Training Accuracy: 98.65%

Training Loss: 0.0413

Validation Accuracy: 95.67%

Validation Loss: 0.1539

# Usage Dependencies:

TensorFlow

NumPy

Matplotlib (for visualization)

# Training:

Clone the repository and load the dataset.

Train the model using the provided architecture.

# Prediction:

Use the trained model to classify new plant leaf images.

# Conclusion

This Plant Disease Detection System is a reliable and efficient tool for detecting plant diseases with high accuracy. It can be deployed as a standalone tool or integrated into agricultural management systems to help mitigate crop loss and improve yield.
