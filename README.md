# ft_linear_regression

<img src="./screenshots/bing-ft_linear_regression.jpg" alt="ML sorting hat" width=55% height=55%></img>

42 school project **ft_linear_regression** could be seen as an entrypoint to the _data science_ branch of 42-school, in the ```outer-circle``` holygraph.
This project is an introduction the field and does not have the pretention to be a fancy data science project.
Machine learning modules, and any module doing the job, are forbidden.

- [Preview](#preview)
- [Subject](#subject)
  - [Dataset](#dataset-to-train)
  - [Mandatory](#mandatory-part)
  - [Bonus](#bonus-part)
- [My solution to linear regression](#my-solution-to-ft_linear_regression)
  - [Usage](#usage) : [venv](#virtual-environment) and [run](#run)
  - [Classes and files](#classes-and-files)
  - [Training](#training) : predicting output values with gradient descent algorithm
- [ft_linear_regression functionalities](#ft_linear_regression-functionalities)

## Preview

Figure 1. Loss surface visualization. Cost function $J(\theta_0, \theta_1)$ is represented on a log scale.

![loss_function](/screenshots/3D_plot.png)

Figure 2. Model Training. Left panel: normalized dataset scatterplot representation, with the line to the predicted value after training. Right panels:  Cost function, $\theta_0$, and $\theta_1)$ are represented through epochs (training iterations).

![model training](/screenshots/params_through_epochs.png)

## Subject

The objective is to implement a ```simple linear regression with a single feature```, _from scratch_. The choice of programming language is free, but should suitable for visualizing data. Using librairies is authorized, except for the ones that does all the work. For example, using python’s ```numpy.polynomial()``` function or ```scikit-learn``` library would be considered as cheating.

### Dataset to train

Monovariate : Car _mileage_ as inputs, car _price_ as output

km|price|
:---:|:---:
240000 |3650|
139800 |3800|
150500 |4400|
...|...|

[data.csv](./data.csv)

### Mandatory Part

A **first program** `predict.py` is predicting the price of a car for a given mileage. The prediction is based on the following model **hypothesis** :

`estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)`

Parameters **thetas** are set to 0 by default, if training did not occur yet.

A **second program** `training.py` is training the model, from a ```data.csv``` train set. According to the hypothesis, both parameters **thetas** are updated with **gradient-descent** algorithm.

The two programs cannot directly communicate. Model parameters issued from training dataset, should be stored and be accessible independently of runtime (**Data persistency**).

### Bonus part

• Plotting the data into a graph to see repartition.

• Plotting the line resulting from linear regression training into the same graph.

• Calculating the precision of the implemented algorithm.

• Any feature that is making sense  

---

## My solution to ft_linear_regression

To implement linear regression from scratch, I chose **Python** language.
Librairies : The power of ```numpy```, a pinch of ```pandas``` and ```matplotlib``` for visualisation.

### Usage

#### Virtual environment

a  _virtual environment_ is necessary so that python and its dependencies are running in an isolated manner, independently from the "system" Python (the host machine).Virtualization with the help of ```Docker``` could be a way to do that in a more complex context. Here, only python installer ```pip```, ```python3``` and few libraries are needed.Thus, ```virtualenv``` is the most straightforward tool ([virtualenv doc.](https://virtualenv.pypa.io/en/latest/) and [python doccs](https://docs.python.org/3/tutorial/venv.html)), and can install a _virtual environment_  from these shell command :

```shell
virtualenv ./venv/
/venv/bin/pip install -r requirements.txt
```

_Makefile_ capabilities were usedto set up _virtual environment_ for **Python**, run programs or clean files. Of course, there is no compilation occuring since **Python** is an interpreted language.

```make``` command will install the virtual environment with dependencies specified in the ```requirements.txt``` file.

```make predict``` to execute the ```predict.py``` program.

```make training``` to execute the ```training.py``` program.

```make flake``` to check for norm with ```flake8```.

```make clean``` to remove ```__pycache__```  and ```.pyc files```.

```make fclean``` to remove the virtual environement after applying the ```clean``` rule.

#### Run

Run with `predict.py` or `training.py`

After, that virtual environment and requirements are installed.

Run with virtual environment python

```shell
venv/bin/python predict.py
```

Otherwise Activate of the virtual environment

```shell
source /venv/bin/activate
```

This will change the shell prompt, to `(venv)` and allow to directly use `venv/bin/*`.
Type only ```pip``` or ```python``` of the ```venv``` with only one word, no need for ```/venv/bin/``` prefix.

```shell
python predict.py
```

### Classes and files

```mermaid
graph TD;
  A[predict.py]-->|instanciate|B[class <br> PredictPriceFromModel];
  C{model  <br>  parameters  <br> persistency}--read-->A[predict.py];
  D[training.py]--instanciates-->E[class  <br> CarPriceDatasetAnalysis];
  E[class  <br> CarPriceDatasetAnalysis]-->F[class  <br> LinearRegressionGradientDescent];
  E[class  <br> CarPriceDatasetAnalysis]--writes-->C{model  <br>  parameters  <br> persistency};
  G{car price <br>  training  <br> dataset}--read-->E[class  <br> CarPriceDatasetAnalysis];
```

### Training

#### Linear regression

The objective is to find a solution to the linear hypothesis model.

For multiple linear regression, the output _response_ ($Y$) linearily depends on a discrete number of $k$ independent variables ($X_j$) also called _predictors_.
With  $\theta_j$, as Weights of the hypothesis for $j$ being the feature index number (from 1 to k).

  **Predicted output** $$y = \theta_0 + \theta_1 * x_1 + \theta_2 * x_2 + ... + \theta_k * x_k$$

In our model, the hypothesis is that _price_ is depending only on _mileage_, therefore $\theta_0$ and $\theta_1$ are the two weigths to be found by our algorithm.

For any x input value, and more specifically any $x_i$, an output predicted value $h(x_i)$ can be calculated with the following linear relationship.

  **Output predicted value** $$h(x_i)=\theta_0 +  \Theta_1 * x_i$$

For any given $x_i$, the calculated predicted value $h(x_i)$ might differ from the real value of $y_i$. These residual are specific to each $x_i$ but also to each $[\theta_0,  \theta_1]$ pair at any step of learning.

#### Gradient descent

The linear-fit relationship to the given dataset is based on the **Sum of Squared Residuals Method**, trying to find the minimize $$\sum_{i=1}^m (h(x_i) - y_i)^2$$ during the learning process,

The **cost function** of the linear regression $J(\theta_0, \theta_2)$, measures the Root Mean Squared error between the predicted value (pred) and true value (y).

**cost function**
$$J(\theta_0, \theta_1) =  \frac{1}{2m} \sum_{k=1}^m (h(x_i)-y_i)^2$$

To implement the ```gradient descent algorithm```, to keeping it simple, the slope of the cost function according to each $\theta$ direction, orientates us toward the minimal cost and tells if that $\theta$ needs to be increased or decreased. In addition to that, it also allows to update the value of that same given $\theta$.

_Partial derivative_ of $J(\theta_0, \theta_1)$ to $\theta_0$
 $$\delta(J(\theta_0, \theta_1))/\delta\theta_0 = \frac{\alpha}{m} \sum_{k=1}^m (h(x_i)-y_i)$$

_Partial derivative_ of $J(\theta_0, \theta_1)$ to $\theta_1$
  $$\delta(J(\theta_0, \theta_1))/\delta\theta_1 = \frac{\alpha}{m} \sum_{k=1}^m (h(x_i)-y_i)x_i$$

 $\alpha$     : Learning Rate of Gradient Descent.

$[\theta_0,  \theta_1]$ pair is updated by decrementing $\alpha * partial derivative$ amount. Linear algebra and ```numpy``` simplifies the equation translation into **python** coding language :

```python
partial_derivative = np.zeros(2)
partial_derivative[0] = np.mean(residual)
partial_derivative[1] = np.mean(np.multiply(self.x, residual))
self.theta -= self.alpha * partial_derivative
```

Developped explanation are found here : [geeksforgeeks.com : gradient descent in linear regression articles](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)

![Formulas for Gradient descent](https://media.geeksforgeeks.org/wp-content/uploads/Cost-Function.jpg)

#### In summary

**Basically, at any step of the learning process:
  The pair $[\theta_0,  \theta_1]$ allows to calculate
  • the cost function J(\theta_0, \theta_2)$ given all the $x_i$ of the trainset.
  • the partial derivative for $theta_0$
  • the partial derivative for $theta_1$
  • update the $[\theta_0,  \theta_1]$ pair accordingly.

### ft_linear_regression functionalities

#### Interactivity and optimisation

In addition to the algoritmic implementation, there is other functional aspects.
At the runtime, `user's input` with `(Y/N)` allow to control training and plotting features.
This interactivity also allow to skip optional features to focus on  **training parameter optimisation** , `learning rate` and `epochs`.

#### Dataset training

* **normalization** of the dataset. The values are in thousands order of magnitude (both _mileage_ and _price_) and needed to be scaled.

* **data persistency** : Subsequently, linear regression parameters has to be stored in a file, so that the model could be further used by the ```predict.py``` program.

* **model metrics** for linear regression analysis and a model accuracy report. [statistics_utils.py](./ft_linear_regression/statistics_utils.py)

#### Plots

Providing many plots, using `matplotlib`.

* 3D plot : **cost function** $J(\theta_0, \theta_1)$, log-scaled. Allows a visual explanation for `minimal cost(s)` point(s) and `gradient descent`.`
* Scatterplot of the trained dataset.
* Same scatterplot with the regression line. The equation, leraning rate and epochs and shown.
* Plot of cost function** $J(\theta_0, \theta_1)$ over epochs, to show the descent to the minimal cost.
* plot of hypothesis parameters $\theta_0$ and $\theta_1$ over epochs.

#### Prediction program

Allows to predict _price_ for a given _mileage_. This relies on persistent data, the model parameters file.

* model parameters are set to value zero, if the persistent model file cannot be read.
* _Exceptions_ are thrown if the user input _price_ is not valid.
