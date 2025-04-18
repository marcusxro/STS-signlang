import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Inspect and filter data
expected_length = 63  # adjust this if your feature vector should be longer/shorter

filtered_data = []
filtered_labels = []

for d, l in zip(data_dict['data'], data_dict['labels']):
    if isinstance(d, (list, np.ndarray)) and len(d) == expected_length:
        filtered_data.append(d)
        filtered_labels.append(l)

# Convert to numpy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
