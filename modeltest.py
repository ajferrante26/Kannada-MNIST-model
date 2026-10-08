import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load model
model = tf.keras.models.load_model('kannada_mnist_dense.h5')

# Load test data
test = pd.read_csv('test.csv')

feature_cols = [c for c in test.columns if c.startswith('pixel')]
if len(feature_cols) == 0:
    feature_cols = [c for c in test.columns if c.lower() not in ['id', 'imageid']]

X_test = test[feature_cols].values.astype('float32') / 255.0
print('Test shape:', X_test.shape)

# Predict
probs = model.predict(X_test, batch_size=256, verbose=1)
preds = np.argmax(probs, axis=1)

# Build submission
try:
    sample = pd.read_csv('sample_submission.csv')
    submission = sample.copy()
    label_col = None
    for c in submission.columns:
        if c.lower() in ['label', 'class', 'target', 'category']:
            label_col = c
            break
    if label_col is None:
        label_col = submission.columns[1]
    submission[label_col] = preds
except FileNotFoundError:
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(preds) + 1),
        'Label': preds
    })

submission.to_csv('submission.csv', index=False)
print(submission.head())
print(submission.shape)
