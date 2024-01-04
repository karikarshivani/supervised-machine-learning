import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

colors = ["Red", "Blue", "Green", "Yellow", "Pink", "Purple", "Orange"]

training_data = pd.read_csv("training_dataset.csv") # Pending import
training_data.head()

testing_data = pd.read_csv("testing_dataset.csv") # Pending import
testing_data.head()

encoder = LabelEncoder()
encoder.fit(training_data["Colour Scheme"])
training_data["Colour Scheme"] = encoder.transform(training_data["Colour Scheme"])
testing_data["Colour Scheme"] = encoder.transform(testing_data["Colour Scheme"])


