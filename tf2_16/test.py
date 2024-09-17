# What version of Python do you have?
import sys
import uproot
import ROOT # need to: apt install -y libgl1-mesa-glx
import numba as nb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch
import mplhep as hep
import tensorflow as tf

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set device to GPU if available, otherwise use CPU
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'
print(f"Using device: {device}")

# Test pandas
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# Test matplotlib
plt.figure()
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
plt.title("Matplotlib Test")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Test scikit-learn
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Test XGBoost
xgb_classifier = xgb.XGBClassifier(objective="multi:softmax", num_class=3)
xgb_classifier.fit(X_train, y_train)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# Test TensorFlow
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, verbose=0)
tf_predictions = np.argmax(model.predict(X_test), axis=1)
tf_accuracy = accuracy_score(y_test, tf_predictions)
print(f"TensorFlow Accuracy: {tf_accuracy}")

# Test Keras Tuner
def build_model(hp):
    model = Sequential([
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(4,)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=3, directory='keras_tuner', project_name='test_tuning')
tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Test mplhep
df = pd.DataFrame({'A': np.random.randn(1000), 'B': np.random.randn(1000)})
plt.figure()
plt.hist(df, bins=30, alpha=0.5, label=df.columns)
plt.legend(loc='upper right')
plt.title("mplhep Test")
plt.show()

print("Finish!")
