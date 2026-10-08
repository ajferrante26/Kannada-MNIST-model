# ============================================================
# Kannada MNIST - Dense Neural Network
# No CNN or convolution layers
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
# If using Kaggle Notebook, change to the correct input path.
# Example:
# train = pd.read_csv('/kaggle/input/kannada-mnist/train.csv')
# test = pd.read_csv('/kaggle/input/kannada-mnist/test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Find the label column
label_col = None
for c in train.columns:
    if c.lower() in ['label', 'class', 'target']:
        label_col = c
        break

if label_col is None:
    raise ValueError('No label column found in train.csv')

# Find pixel columns
feature_cols = [c for c in train.columns if c.startswith('pixel')]

# Fallback if pixel columns are named differently
if len(feature_cols) == 0:
    feature_cols = [
        c for c in train.columns
        if c != label_col and c.lower() not in ['id', 'imageid']
    ]

print(f'Label column: {label_col}')
print(f'Number of features: {len(feature_cols)}')

# ------------------------------------------------------------
# 2. Preprocess the data
# ------------------------------------------------------------
X = train[feature_cols].values.astype('float32') / 255.0
y = train[label_col].values.astype('int64')

X_test = test[feature_cols].values.astype('float32') / 255.0

print('Train shape:', X.shape)
print('Test shape:', X_test.shape)
print('Class counts:', np.bincount(y))

# Optional: visualize a few training images
# This assumes 28x28 grayscale images.
if X.shape[1] == 784:
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(str(y[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 3. Train and validation split
# ------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=SEED,
    stratify=y
)

# ------------------------------------------------------------
# 4. Build the dense neural network
# ------------------------------------------------------------
# Required:
# - At least one hidden layer
# - ReLU activation in hidden layers
# - Softmax activation in the output layer
# - No CNN layers

model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------
# 5. Train the model
# ------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2
)

# ------------------------------------------------------------
# 6. Evaluate the model
# ------------------------------------------------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation loss: {val_loss:.4f}')
print(f'Validation accuracy: {val_acc:.4f}')

y_val_pred = np.argmax(model.predict(X_val), axis=1)

print(classification_report(y_val, y_val_pred, digits=4))

cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Plot training history
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Predict the Kaggle test set and create submission.csv
# ------------------------------------------------------------
test_pred_probs = model.predict(X_test)
test_pred = np.argmax(test_pred_probs, axis=1)

try:
    sample = pd.read_csv('sample_submission.csv')
    submission = sample.copy()

    # Find the label column in the sample submission
    label_sub_col = None
    for c in submission.columns:
        if c.lower() in ['label', 'class', 'target', 'category']:
            label_sub_col = c
            break

    if label_sub_col is None:
        label_sub_col = submission.columns[1]

    submission[label_sub_col] = test_pred

except FileNotFoundError:
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(test_pred) + 1),
        'Label': test_pred
    })
    print('sample_submission.csv not found. Using ImageId,Label format.')

submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')
print(submission.head())

# Optional: save the trained model
model.save('kannada_mnist_dense.h5')
