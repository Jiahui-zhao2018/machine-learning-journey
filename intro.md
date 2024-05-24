# Introduction to Machine Learning

>This is a personal summary of [ML-For-Beginners](https://github.com/kakaichou/ML-For-Beginners)

1. What is the goal of the model?
2. What is the definition of the model?
3. What data structure do we need to train and test the model?
4. How to manipulate the training data and visualize predicted results?
5. How to train the model?
6. How to evaluate the predicted results?


## Table of Contents

- Linear Regression

- Logistic Regression

- Classification

- Clustering

# Linear Regression

## Goal

The goal of a linear regression exercise is to be able to plot a line to:

- **Show variable relationships.** Show the relationship between variables

- **Make predictions.** Make accurate predictions on where a new datapoint would fall in relationship to that line.

## Model

For multiple linear regression with $p$ independent variables, the model is:

$$
\begin{align*}
y & = \beta_0 + \sum_{i=1}^p{\beta_i}x_i+\epsilon \\
  & = X\beta + \epsilon  
\end{align*}
$$

Where $\epsilon$ is the error term.  

The objective is to minimize the sum of the squared residuals(SSR):
$$
SSR = \sum_{i=1}^n(y_i-\beta_0-\beta_1x_{i1}-\beta_2x_{i2}-...-\beta_px_{ip})^2
$$  

To find the parameters that minimize the SSR, take the partial derivatives of SSR to each parameter and set them to 0, the results in a system of linear equations known as the normal equations:
$$
(X^TX)\beta = X^Ty
$$
Solving these equations gives the estimated values of the parameters $\hat{\beta}=(X^TX)^{-1}X^Ty$.

## Data Preparation

We use `Pandas` to analyze and prepare the training data:

```py
import pandas as pd
df = pd.read_csv('path/to/your/csv')
```

Use `head()` to view the first $n$ rows

```py
n = 5
df.head(n)
```

Create a new Pandas dataframe with the existing one:
```py
new_df = pd.dataframe({'key1':df['key1'], 'key2':df['key2'], ...})
```

select data
```py
# Select the data containing a given string in a certain column
df = df[df['key1'].str.contains('str1', case=True, regex=True)]
```

## Data Structure

Linear Regression expects a 2D-array as an input, where each row of the array corresponds to a vector of input features. If you have only one input - we need an array with shape $N×1$, where $N$ is the dataset size.


## Training

We use the `Scikit-learn` library to train our Linear Regression model.

```py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Separate input values and the expected output into separate numpy arrays:
X = df['input_key'].to_numpy().reshape(-1,1)
y = df['label']

# Split the data into train and test datasets, so that we can validate our model after training:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the actual Linear Regression model 
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

## Evaluation and Visualization

To see how accurate our model is, we can predict on a test dataset, and then measure how close our predictions are to the expected values.

```py
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
We can also plot the test data together with the regression line to better see how regression works in our case.

We use `Matplotlib` to create some basic plots to display the dataframe.

```py
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

# Logistic Regression

## Goal

Logistic regression is used to discover patterns to predict binary categories.

Logistic regression will give more accurate results if you use more data, a small dataset is not optimal for this task.

## Model

Logistic regression relies on the concept of 'maximum likelihood' using sigmoid functions. A 'Sigmoid Function' on a plot looks like an 'S' shape. It takes a value and maps it to somewhere between 0 and 1. Its curve is also called a 'logistic curve'. Its formula looks like this:

$$
f(x) = \frac{L}{1+e^{-k(x-x_0)}}
$$

Where the sigmoid's midpoint finds itself at x's 0 point, L is the curve's maximum value, and k is the curve's steepness.  

If the outcome of the function is more than 0.5, the label in question will be given the class '1' of the binary choice. If not, it will be classified as '0'.

## Data Preparation

Select the variables you want to use in your classification model and split the training and test sets calling `train_test_split()`:

```py
from sklearn.model_selection import train_test_split

X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
y = encoded_pumpkins['Color']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```