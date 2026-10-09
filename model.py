# ============================================================
# Kannada MNIST - Dense Neural Network
# No CNN or convolution layers
# ============================================================
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; figures saved to disk
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# Output directory for artifacts
# ------------------------------------------------------------
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(ARTIFACT_DIR, 'history.json')

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# History logger callback
# Writes history.json incrementally after every epoch
# ------------------------------------------------------------
class HistoryLogger(callbacks.Callback):
    def __init__(self, path=HISTORY_PATH):
        super().__init__()
        self.path = path

    def on_train_begin(self, logs=None):
        with open(self.path, 'w') as f:
            json.dump({}, f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            with open(self.path) as f:
                h = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            h = {}
        for k, v in logs.items():
            h.setdefault(k, []).append(float(v))
        with open(self.path, 'w') as f:
            json.dump(h, f, indent=2)

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
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

# Optional: visualize a few training images and save for the report
if X.shape[1] == 784:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.set_title(str(y[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'sample_images.png'), dpi=150)
    plt.close()

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
print(f'Train subset: {X_train.shape}, Val subset: {X_val.shape}')

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

# Save architecture summary to a text file for the report
with open(os.path.join(ARTIFACT_DIR, 'model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda line: f.write(line + '\n'))

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
    callbacks=[early_stop, HistoryLogger(HISTORY_PATH)],
    verbose=2
)

# ------------------------------------------------------------
# 6. Evaluate the model
# ------------------------------------------------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation loss: {val_loss:.4f}')
print(f'Validation accuracy: {val_acc:.4f}')

y_val_pred = np.argmax(model.predict(X_val), axis=1)

report = classification_report(y_val, y_val_pred, digits=4)
print(report)
with open(os.path.join(ARTIFACT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

cm = confusion_matrix(y_val, y_val_pred)

# Per-class accuracy from the diagonal
per_class_acc = cm.diagonal() / cm.sum(axis=1)
with open(os.path.join(ARTIFACT_DIR, 'per_class_accuracy.json'), 'w') as f:
    json.dump({str(i): float(v) for i, v in enumerate(per_class_acc)}, f, indent=2)
print('Per-class accuracy:', np.round(per_class_acc, 4))

# Top confused pairs (off-diagonal counts)
off = cm.copy()
np.fill_diagonal(off, 0)
top_pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0]
print('\nTop 5 confused pairs (true -> predicted, count):')
confused_lines = []
for i, j in top_pairs[:5]:
    line = f'  {i} -> {j}: {int(off[i, j])}'
    print(line)
    confused_lines.append(line)
with open(os.path.join(ARTIFACT_DIR, 'confused_pairs.txt'), 'w') as f:
    f.write('\n'.join(confused_lines))

# Confusion matrix figure
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(range(10)), yticklabels=list(range(10))
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

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
plt.savefig(os.path.join(ARTIFACT_DIR, 'training_curves.png'), dpi=150)
plt.close()

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

# ------------------------------------------------------------
# 9. Save model and training summary
# ------------------------------------------------------------
model.save('kannada_mnist_dense.h5')

with open(HISTORY_PATH) as f:
    h = json.load(f)

best_epoch = int(np.argmin(h['val_loss'])) + 1
summary = (
    f"Epochs run:           {len(h['loss'])}\n"
    f"Best epoch:           {best_epoch}\n"
    f"Best val_loss:        {min(h['val_loss']):.4f}\n"
    f"Best val_accuracy:    {max(h['val_accuracy']):.4f}\n"
    f"Final train_loss:     {h['loss'][-1]:.4f}\n"
    f"Final train_accuracy: {h['accuracy'][-1]:.4f}\n"
    f"Final val_loss:       {h['val_loss'][-1]:.4f}\n"
    f"Final val_accuracy:   {h['val_accuracy'][-1]:.4f}\n"
)
print()
print('=== Training Summary ===')
print(summary)

with open(os.path.join(ARTIFACT_DIR, 'training_summary.txt'), 'w') as f:
    f.write(summary)
