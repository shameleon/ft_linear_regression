
# ft_linear_regression


[Data persistence](### Data persistence)

## Subject requirements

## Linear regression

Linear regression is a```supervised machine-learning algorithm```

![ML linear regression (geeksforgeeks.org)](https://www.geeksforgeeks.org/ml-linear-regression/)

"A linear regression model can be trained using the optimization algorithm gradient descent by iteratively modifying the model’s parameters to reduce the mean squared error (MSE) of the model on a training dataset."

## Normalization training dataset

Normalization generally refers to processes that achieve scales between zero and one, while standardization uses a principle called the standard deviation to describe the distribution of the data points. 

![which-models-require-normalized-data](https://towardsdatascience.com/which-models-require-normalized-data-d85ca3c85388)

If you train a linear regression without previous normalization, you can’t use the coefficients as indicators of feature importance. If you need to perform feature importance (for example, for dimensionality reduction purposes), you must normalize your dataset in advance, even if you work with a simple linear regression.

## Data persistence

As needed, between the execution of the two programs ```training``` a linear regression model will save data in a file so that ```prediction```  executions
Persistence  storing data in a way that it will persist beyond the run-time of your program.

### pickle module

