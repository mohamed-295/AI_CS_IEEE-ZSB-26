# Supervised vs Unsupervised Learning
### Supervised Learning
The model is trained on labeled data consisting of independent features (inputs) and a dependent feature (the target [output]).
* *Regression:* The output is a continuous number. 
* *Classification:* The output falls into fixed categories.
### Unsupervised Learning
The model is given data with no output labels and must find hidden patterns. 
* *Clustering:* Grouping similar data points together. 
* *Dimensionality Reduction:* Reducing the number of input variables while retaining essential information.


# Linear Regression & Gradient Descent

Linear Regression aims to find best fit line "linear" => ($y = mx + c$) that models the relationship between independent features and a continuous target output. It can be done with a single variable or multiple variables. 

- **Cost Function (MSE):** To determine if a line is the "best fit" we measure the error between our model's predicted points (y_pred) and the actual real points (y). The standard way to do this is by squaring those distances and finding the average. The goal is to minimize this Cost Function. 
* **Gradient Descent:** This is an optimization algorithm used to iteratively update our line's coefficients (slope and intercept) until we reach the **Global Minima** (the point where the cost function is at its lowest possible value). 
* **Learning Rate ($\alpha$):** A hyperparameter in Gradient Descent that determines the "step size" taken towards the global minima. If $\alpha$ is too small, training takes forever; if it's too large, the model might overshoot the minimum and fail to converge.

# L1 & L2 Regularization (Ridge and Lasso)

When a model strictly memorizes the training data perfectly (including all the noise and outliers), it results in a curve that is highly steep and chaotic.

* **Overfitting:** The model performs good on training data (Low Bias) but fails terribly on unseen test data (High Variance). 
* **Underfitting:** The model is overly simplistic and performs terribly on both training and test data (High Bias, High Variance). 
* **Ridge Regression (L2 Regularization):** Solves overfitting by adding a penalty term to the cost function equal to  (**$\lambda \times (\text{slope})^2$**). This shrinks the slope values and makes the line less steep, ensuring a more generalized model that reacts better to new, unseen data. 
* **Lasso Regression (L1 Regularization):** Adds a penalty term equal to **$\lambda \times |\text{slope}|$**. Aside from preventing overfitting, Lasso excels at **Feature Selection**. If a dataset has useless features, Lasso will shrink their specific slopes down to absolute zero, practically removing them from the equation.