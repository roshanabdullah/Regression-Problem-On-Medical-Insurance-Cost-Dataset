# import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
insurance

# Lets try one-hot encoding to that all our Dataframe it's all numbers
insurance_one_hot = pd.get_dummies(insurance)
insurance_one_hot.head()

# Create X and y values (features and labels)
X = insurance_one_hot.drop("charges", axis=1) # drop charges as X are all features
y = insurance_one_hot["charges"] # get charges as this is the output

# View X
X.head()

# View Y
y.head()

# Create training and test sets
from sklearn.model_selection import train_test_split # splits the training and test sets randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X), len(X_train), len(X_test)

# Build neural network

# take X_train and y_train and learn their relationships
tf.random.set_seed(42)

#1. Create a model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

#2. Compile the Model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"]
                        )

#3. Fit the model
insurance_model.fit(X_train, y_train, epochs=100)

# Check the results of the insurance model on the test data
insurance_model.evaluate(X_test, y_test)

"""# Right now it looks like our model is not performing well, lets try and improve it...

To (try) improve our model, we will run 2 experiments:
1. Add an extra layer with more hidden units and use the Adam optimizer
2. Same as above but this time Train for longer (200 epochs)
3. (insert your own experiments here)

"""

# set random seed
tf.random.set_seed(42)

# Create the model
insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

# Fit the model
insurance_model_2.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the larger model
insurance_model_2.evaluate(X_test, y_test)

# Set random seed
tf.random.set_seed(42)


# Create the model
insurance_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

# fit the model
history = insurance_model_3.fit(X_train, y_train, epochs=200)

# Evaulate our third model
insurance_model_3.evaluate(X_test, y_test)

# Plot history (also know as loss curve or a training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

"""**Question** How long should you train for?

It depends on the problem you are working on. However many people have asked this question before ... so TensorFlow has a solution! Its called the [EarlyStopping Callback], which is a TensorFlow component you can add to your model to sotp training once it stops improving a certain metric

# Preprocessing data (normalization and standardization)

In terms of scaling values, neural networks tends to prefer normalization.

If youre not sure on which to use, you could try both and see which performs better
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Read in the insurance dataframe
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
insurance

"""To prepare our data, we can borrow a few classes from Scikit-Learn."""

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children", ]),#turn all values in these columns between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# create x and y values
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # test size is set test data to 20%

# Fit the column transformer to our training data
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# What does our data look like now
X_train.loc[0]

X_train_normal[0]

X_train.shape, X_train_normal.shape

"""Our data has been normalized and one hot encoded. Now lets build a neural network model on it and see how it goes."""

# Build a neural network model to fit on our normalized data
tf.random.set_seed(42)

# Create the model
insurance_model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_4.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"]
                          )

# Fit the model
insurance_model_4.fit(X_train_normal, y_train, epochs=100)

# Evaluate our insurance model trained on normalized data
insurance_model_4.evaluate(X_test_normal, y_test)

X["age"].plot(kind="hist") # plotting our non-normalized dataset to check how much it scales commonly

X["bmi"].plot(kind="hist")  # plotting our non-normalized dataset to check how much it scales commonly

X["children"].value_counts()  # plotting our non-normalized dataset to check how much it scales commonly

