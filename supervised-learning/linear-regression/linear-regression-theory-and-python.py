# <h1>INTRODUCTION TO THE NOTEBOOK </h1>
# 
# 
# 
# <h2>Aim</h2>
# 
# In this notebook, we will explore the principal characteristics of the linear regression models. This document is designed to emphasize technical details over the typical step-by-step procedure for implementing a regression model. Consequently, the organization of sections does not follow the traditional sequence. 
# 
# Some limitations of this notebook:
# 
# * We didn´t split data into training and test. 
# * No cleaning or exploration of the data was carried out.
# * The assumptions of the model we not tested.
# 
# A complementary document will be provided, featuring a practical example to cover it thoroughly.
# 
# Please feel free to suggest any corrections, modifications, or improvements. **Your feedback is greatly appreciated!**.
# 
# <h2>Programming Language: Python</h2>
# 
# The code in the following sections is developed using Python (v. 3.11.8). The versions of the packages used are:
# 
# * **Pandas**: '2.1.4' 
# * **scikit-learn**: '1.4.2' 
# * **Matplotlib**: '3.8.0'
# * **Numpy**: '1.26.4'
# * **Statsmodels**: '0.14.1'

# In[3]:


import pandas as pd                                  # Read csv
from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # interaction effect, no linear models
from sklearn.linear_model  import Ridge, RidgeCV, Lasso, LassoCV # regularizationy, best value of alpha Ridge, reduce features, best value of alpha Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor
import matplotlib
from sklearn.preprocessing import StandardScaler # standardization
import matplotlib.pyplot as plt                  # plot
import numpy as np
import statsmodels.api as sm


# Check Python and packages version.
import sys
from sklearn import __version__ as sklearn_version # Only needed for sklearn

versions = {
    "Python": sys.version.split(" ")[0],  # Simplifying to just the version number
    "Pandas": pd.__version__,
    "scikit-learn": sklearn_version,
    "Matplotlib": matplotlib.__version__,
    "Numpy": np.__version__,
    "Statsmodels": sm.__version__
}

#print(versions)


# <h2>Database</h2>
# 
# Throughout this notebook, we will use the "Red Wine Quality" database. You can find a complete description of this dataset and download it by following this link: [Red Wine Quality Dataset Description](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).
# 
# * All the variable in this database are quantitative.
# * The target variable is "quality", which ranges from 0 to 10.
# 
# Now, we are going to upload the data and separate target from features. 

# In[4]:


# Read csv and see the chaacteristics
df = pd.read_csv('winequality-red.csv')
print("Number of rows:", len(df))
print(df.head(5))

# Separate target and features:

df_target   = df[['quality']]      # Use doble [[ ]] to avoid transform in serie (because is only one column)
df_features = df.drop('quality', axis=1)

print("\n")
print("The compleate df has", len(df),          "cases and", df.shape[1],          "columns")
print("df_target has",        len(df_target),   "cases and", df_target.shape[1],   "column")
print("df_features has",      len(df_features), "cases and", df_features.shape[1], "columns")


# <br>
# 
# **Summary of the data**
# 
# We have 1,599 entries (cases,rows), one target variable ("quality") and 11 features.
# 
# <br>
# <br>
# 
# <h2>Some clarifications</h2>
# 
# Along this document, different models will be studied. The table bellow shows a summary of the nomenclature used in Python. I hope this will make it easier to keep track of the code.
# 
# | Model number | Model name                           | Target name | Target Variable | Feature/s name    | Feature/s Variable      | Model created    | Fit model | Predicted value |
# |--------------|--------------------------------------|-------------|-----------------|-------------------|-------------------------|------------------|-----------|-----------------|
# | 0            | Null model                           | Quality     | df_target       | Mean              |                         |                  | model0    | model0_y        |
# | 1            | Simple Linear Regression             | Quality     | df_target       | Alochol           | df_feature_one          | regression1      | model1    | model1_y        |
# | 2            | Multiple Linear Regression           | Quality     | df_target       | All               | df_feature              | regression2      | model2    | model2_y        |
# | 3            | Interaction                          | Quality     | df_target       | All + interaction | df_features_interaction | regression3      | model3    | model3_y        |
# | 4            | Polynomial                           | Quality     | df_target       | All + polynomial  | df_features_polynomial  | regression4      | model4    | model4_y        |
# | 5            | Ridge                                | Quality     | df_target       | All               | df_feature              | ridge_parameters | model5    | model5_y        |
# | 6            | Lasso                                | Quality     | df_target       | All               | df_feature              | lasso_parameters | model6    | model6_y        |
# | 7            | Multiple Linear Regression Stanrized | Quality     | df_target_stand | All standarized   | df_features_stand       | regression7      | model7    | model7_y        |

# <h1>1. Introduction to Linear Regression</h1><a id="section_1"></a>
# 
# **Linear regression** is a machine learning algorithm classified under "supervised learning" techniques.
# 
# The **aim** of linear regression algorithm is to predict one variable (VD, also known as *target* or *y*) using one or more features (VI, *predictors* or *$x_i$*).
# 
# There are two type of models depending on the number of features involve:
# 
# * **Simple Linear Regression**. The model use only one feature to predict the target:
# 
# <div style='font-size: 16px;'>
# $$
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \epsilon
# $$
# </div>
# 
# * **Multiple Linear Regression**. This model use two or more features to predict the target.
# 
# <div style='font-size: 16px;'>
# $$
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 +  \hat{\beta}_2 x_2 + ... + \hat{\beta}_n x_n + \epsilon
# $$
# </div>
# 
# Where:
# 
# * $ \hat{y} $: Is the target we predict.
# * $ \hat{\beta}_0 $: Is the *intercept* or *bias*.
# * $ \hat{\beta}_1, \hat{\beta}_2, ... $: Are the *coefficients* (*weights*, *effect*) associated with each feature. Is the effect of one feature on the target.
# * $ \epsilon $: Represents the error term.
# 
# 
# **Note**: When discussing estimations (like parameter estimations or model residuals) or predictions, we use the hat notation, as in $\hat{y}$.

# <h2>A. Model 1: Simple Linear Regression Model</h2><a id="section_1_A"></a>
# 
# This model uses only one feature to predict the "quality" of the wine, which is our target variable stored in *df_target*. 
# 
# 
# The predictor we will use is the "alcohol" level, currently housed within *df_features* along with other features. To simplify our analysis, we will first extract this specific feature into a new object dedicated solely to it.

# In[5]:


# Select only the feature alcohol
df_feature_one = df_features[['alcohol']]

# Print results
print("Target quality \n", df_target.sort_index().head(5), "\n")
print("Feature alcohol \n", df_feature_one.sort_index().head(5))


# We can see that the first wine [0] have a real "quality" of 5 and an "alcohol" level of 9.4.
# 
# <br>
# 
# Next, we'll construct the model to establish the relationship between "quality" and "alcohol" content. Here's the step-by-step process:
# 
# * **Create the model**. In this case, Linear Regression model.
# * **Fit the model**. Train the model with our data. During this process, the model will adjust its parameters to best fit the relationship between "quality" and "alcohol" content. As a result, we'll obtain the coefficients and the intercept.
# * **Display Coefficients and Intercept**: After fitting the model, we'll showcase the numerical values for the intercept and coefficients. These values provide insights into the baseline "quality" (intercept) and the magnitude of influence (coefficient) that "alcohol" level has on the "quality" of the product.

# In[6]:


# Create SLRM and fit it
regression1 = LinearRegression()

# Fit the model
model1 = regression1.fit(df_feature_one.sort_index(),   # Features first then target
                         df_target.sort_index()) 

# Show the results (round to three decimals):
print("Intercept:",     model1.intercept_[0].round(3))
print("Coefficients :", model1.coef_[0][0].round(3))


# Whith this information, we can construct the equation of the model:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 1.875 + 0.361 · x_1 + \epsilon
# $$
# </div>
# 
# The error $ \epsilon $ epresents the disparity between the actual value of  $ y $ and the predicted value $ \hat{y} $. We can only ascertain this value when we possess the actual result, not during prediction. We display the error only here but it is present in all equations we are going to see in which we made predictions.
# 
# Nevertheless, we can substitute the feature value into the equation. For our initial scenario, this value is:

# In[7]:


print("Value of alcohol feature for our first case :", df_feature_one.iloc[0:1])


# The equation for the first case is:
# 
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = 1.875 + 0.361 · 9.4 
# $$
# </div>
# 
# So, the predicted value of our first case is:
# 
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 1.875 + 0.361 · 10 \approx 5.268
# $$
# </div>
# 
# We can achieve this using our script:

# In[8]:


model1_y = pd.DataFrame(model1.predict(df_feature_one.sort_index()), 
                      index=df_feature_one.index, 
                      columns=['Simple Model'])

print("Quality predicted value for the frst case:", model1_y.sort_index().round(3).head(1))


# Now, we are going to plot our model, putting in relation the predicted and real values of the "quality" of wine. The elements are:
# 
# * **x-axis**: Real value of the target.
# * **y-axis**: Predicted value of the target.
# * **Dots**: Represent each wine "quality" (predicted vs real). The more filled dots are, the more cases concentrated on that relation.
# * **Predicted line**. Dashed red line. Represents the predictions if the model was perfect. The closer the dots to the line, the better predictions are made.
# * **Axis limits**. Ranging between 0 to 10, which are the minimum and maximum theoretical values of the "quality".
# 
# The interpretation that we can make is:
# 
# * The real values of the target range between 3 and 9, while the predicted values range between 5 and 7 (approximately). So not all the variability is captured by the model.
# * Dots are closer to the predicted line in the medium values. This means that our model is not good at predicting when the "quality" of wine is lower or higher. However, overall, the model doesn't predict very well.
# 
# You can see the code to make this plot and the output of it:

# In[9]:


plt.figure(figsize=(8, 6))
plt.scatter(df_target.sort_index(),     # Real value
            model1_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model1_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)
plt.show()


# <h2>B. Model 2: Multiple Linear Regression Model</h2><a id="section_1_B"></a>
# 
# This model uses more than one feature to predict the "quality" of the wine, which is our target variable stored in df_target.
# 
# The predictors we will use are all the available features (11 variables), stored in df_features.
# 
# The steps are identical to those in model 1.

# In[10]:


# Create MLRM and fit it
regression2 = LinearRegression()

# Fit the model
model2 = regression2.fit(df_features.sort_index(), 
                         df_target.sort_index()) 

# Show the equation values:
print("Intercept :",    model2.intercept_ [0].round(3))

coefficient_dict = dict(zip(df_features.columns,
                            model2.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# Whith this information we can build the equation of the model:
# 
# <br>
# 
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 21.956 + 0.025 · x_1 - 1.084 · x_2 - 0.183 · x_3 + 0.016 · x_4 - 1.874 · x_5 + 0.004 · x_6 - 0.003 · x_7 - 17.881 · x_8 - 0.414 · x_9 + 0.916 · x_{10} + 0.276 · x_{11}
# $$
# </div>
# 
# We can see that we have negative and positive values in the coeficients. We interpret them as follows:
# 
# * **Positive**: A high value in that feature increases the value of the target.
# * **Negative**: A high value in that feature decreases the value of the target.
# 
# If we replace the features for our first case, we obtain this:

# In[11]:


print("Value of all features for our first case :", df_features.iloc[0:1].transpose().round(3))


# For the first case the final equation looks like:
# 
# <br>
# 
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 21.956 + 0.025 · 7.4 - 1.084 · 0.7 - 0.183 · 0 + 0.016 · 1.9 - 1.874 · 0.076 + 0.004 · 11 - 0.003 · 34 - 17.881 · 0.998 - 0.414 · 3.510 + 0.916 · 0.560 + 0.276 · 9.4
# $$
# </div>
# 
# And the "quality" predicted value :
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} \approx 5.021
# $$
# </div>
# 
# We can obtain this result using our script (discrepancies might be due to differences in decimal precision)

# In[12]:


model2_y = pd.DataFrame(model2.predict(df_features.sort_index()), 
                        index=df_features.index, 
                        columns=['Multiple Model'])

print("Quality predicted value for the frst case:", model2_y.sort_index().round(3).head(1))


# <br>
# 
# As before, we represent the relation between prediction and real value using a plot. We plot the result from model 1 and model 2 side by side to compare both models.
# 
# There are no large differences between both models. Model 2 performe a slightly better than model 1. We can affirm that because the dots are more close to the red line in model 2 (look when real wine "quality" is 5).

# In[13]:


plt.figure(figsize=(12, 6))


# Plot model 1
plt.subplot(1, 2, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model1_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model1_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Simple Regression: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)



# Plot model 2
plt.subplot(1, 2, 2)

plt.scatter(df_target.sort_index(),     # Real value
                model2_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model2_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


plt.show()


# <h1>2. Features modifications</h1><a id="section_2"></a>
# 
# <h2>A. Interacction effect</h2><a id="section_2_A"></a>
# 
# Sometimes, individual features may not exhibit a significant effect on the target variable by themselves, but their combination does. We can account for this type of effect by including an interaction term in the model.
# 
# Including interaction terms in the model allows for the possibility that the relationship between predictors and the target variable is not additive ($ \hat{\beta}_1 \cdot x_1 + \hat{\beta}_2 \cdot x_2 $), but varies depending on the levels of other predictors ($ \hat{\beta}_3 \cdot x_1 \cdot x_2 $).
# 
# Imagine we have an original model with two features: $x_1$ and $x_2$. As explained in the introduction, the model will take this form:
# 
# <br>
# 
# <div style='font-size: 14px;'>
# $$
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + \epsilon
# $$
# </div>
# 
# If we suspect there is an interaction between these two features, we can add the interaction effect as follows:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + \hat{\beta}_3 x_1 x_2 + \epsilon
# $$
# </div>
# 
# Some relevant points:
# 
# * We can include as many interaction effects as we want if they are interesting. However, we should consider the relevance of including them because we run the risk of overfitting our model.
# * An interaction effect could involve more than two features. For example, ($ \hat{\beta}_3 · x_1 · x_2 · x_3 $). If you decide this is interesting in your case, be cautious about overfitting and the difficulty of interpreting the interaction.
# * It is not necessary to include interactions for all the features involved in the model. If we have a model with three features and we think only the interaction between $ x_1 $ and $ x_2 $ is ineresting, then we can create a model like this: 
# 
# <br>
# <div style='font-size: 14px;'>
# $$ 
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + \hat{\beta}_3 x_3 + \hat{\beta}_4 x_1 x_2 + \epsilon 
# $$
# </div>
# 
# Now, let's learn how to implement this in Python.
# 
# As before, our target variable is "quality". For predictors, we'll include all the features, along with the interaction between "free sulfur dioxide" and "alcohol".
# 
# First, we'll create the interaction term:
# 

# In[14]:


interaction = PolynomialFeatures(
    degree=2,                    # How many features combinations we want to create
    include_bias=False,          # Exclude the bias term 
    interaction_only=True        # Only include interaction effects (not x^2, x^3, etc.)
    )


# <br>
# 
# Now, we have to transform and fit the interaction between "free sulfur dioxide" and "alcohol":

# In[15]:


features_interaction = interaction.fit_transform(df_features[['free sulfur dioxide', 'alcohol']])

# This is not necessary but is good to see the results. Transform the array into a dataframe:
features_interaction = pd.DataFrame(features_interaction,
                                      columns = interaction.get_feature_names_out(['free sulfur dioxide', 'alcohol']),
                                      index=df_features.index  # This line ensures the index is carried over. Example, index 10 doesn´t should appear
                                   )
print(features_interaction.sort_index().head(1))


# <br>
# 
# As we observed, the interaction term includes both the simple features (since degree = 2 encompasses degree = 1, which represents the simple effects) and the interaction itself. We are specifically interested in the interaction between "free sulfur dioxide" and "alcohol". Therefore, we extract only that column and add it to our original features DataFrame:

# In[16]:


interaction_column = features_interaction[['free sulfur dioxide alcohol']].sort_index()
print(interaction_column.sort_index().head(5))


# <br>
# We add the interaction column to our original features DataFrame. To demonstrate that this DataFrame has been modified, we will create a new DataFrame to display the modified version.

# In[17]:


df_features_interaction = df_features.join(interaction_column, how='outer')

print(df_features_interaction.sort_index().head(5))


# <br>
# 
# Now, we have a new DataFrame with all the original features plus one additional column for the interaction term (which is the last column, "free sulfur dioxide alcohol").
# 
# **Important Note**: In this notebook, we are only utilizing the global database without splitting it into train and test sets. However, if we were to conduct such a split, the calculation of the interaction should be performed before splitting the data.
# 
# With our final DataFrame, df_features_interaction, we can proceed to fit the model and obtain the intercept and coefficients:

# In[18]:


# Create MLRM and fit it
regression3 = LinearRegression()

# Fit the model
model3 = regression3.fit(df_features_interaction.sort_index(), 
                         df_target.sort_index()) # Features first then target

# Show the equation values:
print("Intercept :",    model3.intercept_[0].round(3))

coefficient_dict = dict(zip(df_features_interaction.columns,
                            model3.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# Now create the equation of the model:
# 
# <br>
# 
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 21.055 + 0.024 · x_1 - 1.082 · x_2 - 0.170 · x_3 + 0.019 · x_4 - 1.840 · x_5 - 0.018 · x_6 - 0.003 · x_7 - 16.610 · x_8 - 0.420 · x_9 + 0.903 · x_{10} + 0.245 · x_{11} + 0.002 · x_{6} · x_{11}
# $$
# </div>
# 
# If we substitute the features for our first case, we obtain the following:

# In[19]:


# Notice that we use df_features_interaction
print("Value of all features for our first case :", df_features_interaction.iloc[0:1].transpose().round(3))


# <br>
# 
# We replace with the features information:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 21.055 + 0.024 · 7.4 - 1.082 · 0.7 - 0.170 · 0 + 0.019 · 1.9 - 1.840 · 0.076 - 0.018 · 11 - 0.003 · 34 - 16.610 · 0.998 - 0.420 · 3.510 + 0.903 · 0.560 + 0.245 · 9.4 + 0.002 · 103.4 \approx 5.036
# $$
# </div>
# 
# Using Python:

# In[20]:


model3_y = pd.DataFrame(model3.predict(df_features_interaction.sort_index()), 
                        index=df_features_interaction.index, 
                        columns=['Interaction Model'])

print("Quality predicted value for the frst case:", model3_y.sort_index().round(3).head(1))


# <br>
# We are set to examine the relationship between the actual "quality" and the predicted "quality" of our model. We include two plots with two models: multiple regresion and interaction. Both include the 11 original features, but the second model also incorporates an interaction term. Despite these differences in model configuration, our findings indicate that there are no discernible differences in their predictions.

# In[21]:


plt.figure(figsize=(12, 6))


# Plot model 2
plt.subplot(1, 2, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model2_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model1_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)



# Plot model 3
plt.subplot(1, 2, 2)

plt.scatter(df_target.sort_index(),     # Real value
                model3_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model3_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression with interaction term: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


plt.show()


# <h2>B. Nonlinear relationships</h2><a id="section_2_B"></a>
# 
# In linear models, we typically expect the relationship between the target and features to be constant. However, when this assumption doesn't hold true, the relationship is deemed nonlinear. For instance, consider a student preparing for an exam. Generally, more hours of study lead to higher scores. However, this relationship is not linear. Spending one additional hour studying doesn't necessarily result in a score of one additional score on the test, nor does studying for seven hours guarantee an increase on seven points on the score. Instead, the relationship between the target ("score") and the feature ("number of hours of study") is not necessarily linear if not maybe exponential.
# 
# A model that incorporates a nonlinear effect in a feature accounts for this aspect:
# 
# <br>
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 +  \hat{\beta}_2 x_1^2 + ... + \hat{\beta}_n x_n + \epsilon
# $$
# </div>
# 
# Where:
# 
# * $ x_1 $: Represents the feature with a linear effect.
# * $ x_1^2 $: Represents the same feature with a nonlinear association. The exponent $ ^2 $ indicates the polynomial degree.
# 
# <br>
# 
# **Include only nonlinear relationshiph or both?**
# 
# If you believe that the relationship between a feature and the target variable is purely nonlinear, then you might only include the nonlinear term in your model. However, in many cases, including both linear and nonlinear terms can provide a more flexible and accurate representation of the relationship.
# Additionally, including both linear and nonlinear terms can help prevent issues like omitted variable bias, where failing to include relevant variables in the model leads to biased and inconsistent parameter estimates.
# 
# 
# 
# The approach (Python) is similar to that of incorporating interaction effects. We will create the complete model, including nonlinear effect for the "sulfite" feature and then create and fit the model.
# We begin by creating a new feature "$ sulphite^2 $", which represents the square of the "sulfite" feature. This is essentially an interaction between the same feature.
# 
# 

# In[22]:


polynomial = PolynomialFeatures(
    degree=2,                    # We need a interaction with two degrees.
    include_bias=False,          # Exclude the bias term 
    interaction_only=False       # NEW. Include polynomia effects. In our case only x^2 because we are interesting in 2 degrees.
    )


# Now, we have to transform and fit the polynomial effect of "sulphates":

# In[23]:


polynomial_features = polynomial.fit_transform(df_features[['sulphates']].sort_index())

# This is not necessary but is good to see the results. Transform the array into a dataframe:
features_polynomial = pd.DataFrame(polynomial_features,
                                   columns = polynomial.get_feature_names_out(),
                                   index=df_features.index  # This line ensures the index is carried over. Example, index 10 doesn´t should appear
                                   )
print(features_polynomial.sort_index().head(1))


# <br>
# We only need incorporate the $ sulphates^2 $ column. First, we isolete that column and then we add it to the original feature df (we create a new dataframe for this new data)

# In[24]:


# Isolate the polynomial feature
polynomial_column = features_polynomial[['sulphates^2']].sort_index()

# Add to the other features (and create a new features df)
df_features_polynomial = df_features.join(polynomial_column, how='outer')
print(df_features_polynomial.sort_index().head(5))


# <br>
# 
# **Important Note**. As we say in the interaction section, if we are going to split the data into training and testing, then this steps should be done before.
# 
# With our final DataFrame, df_features_polynomial, we can proceed to fit the model and obtain the intercept and coefficients:

# In[25]:


# Create MLRM and fit it
regression4 = LinearRegression()

# Fit the model
model4 = regression4.fit(df_features_polynomial.sort_index(), 
                         df_target.sort_index()) # Features first then target

# Show the equation values:
print("Intercept :",    model4.intercept_[0].round(3))

coefficient_dict = dict(zip(df_features_polynomial.columns,
                            model4.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# 
# <br>
# 
# Now build the equation of the model:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 32.52 + 0.013 · x_1 - 0.967 · x_2 - 0.214 · x_3 + 0.017 · x_4 -1.599 · x_5 + 0.003 · x_6 - 0.003 · x_7 - 28.532 · x_8 - 0.662 · x_9 + 3.640 · x_{10} + 0.260 · x_{11} - 1.565 · x_{10}^2
# $$
# </div>    
# 
# Observing the polynomial coefficient, it becomes evident that it's negative. This signifies that as the levels of "sulfates" increase, the "quality" of the wine tends to decrease. Given its polynomial nature, this decline in "quality" accelerates, indicating a rapid deterioration with higher "sulfates".
# 
# Upon substituting the features in our initial case, the resultant trend emerges as follows:

# In[26]:


# Notice that we use df_features_interaction
print("Value of all features for our first case :", df_features_polynomial.iloc[0:1].transpose().round(3))


# <br>
# 
# We replace our model with the features information case one:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 32.52 + 0.013 · 7.4 - 0.967 · 0.7 - 0.214 · 0 + 0.017 · 1.9 -1.599 · 0.076 + 0.003 · 11 - 0.003 · 34 - 28.532 · 0.998 - 0.662 · 3.510 + 3.640 · 0.560 + 0.260 · 9.4 - 1.565 · 0.314 \approx 4.973
# $$
# </div>
# 
# Using Python:

# In[27]:


model4_y = pd.DataFrame(model4.predict(df_features_polynomial.sort_index()), 
                      index=df_features_polynomial.index, 
                      columns=['Polinomyal Model'])

print("Quality predicted value for the frst case:", model4_y.sort_index().head(1))


# We create again the two plots: multiple regression and regression with polynomical term. Again, no differences are detected.

# In[28]:


plt.figure(figsize=(12, 6))


# Plot model 2
plt.subplot(1, 2, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model2_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model1_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)



# Plot model 4
plt.subplot(1, 2, 2)

plt.scatter(df_target.sort_index(),     # Real value
                model4_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model4_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression with polynomical term: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


plt.show()


# <br>
# <br>
# 
# <h1>3. Optimization Techniques</h1><a id="section_3_A"></a>
# 
# Scikit-learn offers a wide array of estimator options, covering various techniques beyond regression. The image below illustrates the extensive variety included in this package. For more details, you can explore [Scikit-learn estimators](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html).
# <br>
# 
# <div style="text-align:center">
#     <img src="https://scikit-learn.org/stable/_static/ml_map.png" alt="Scikit-Learn estimators" width="700" height="500">
# </div>
# 
# <br>
# 
# **Optimization algorithms** are tools used to find the best solution to a problem, typically aiming to minimize or maximize a function. Key concepts include:
# 
# * **Objective Function**: The function we aim to optimize.
# * **Minimum and Maximum**: We seek the minimum when optimizing parameters to minimize error and the maximum when maximizing profits.
# * **Stopping Conditions**: Criteria indicating when to cease the search for the optimal solution. For instance, fixed iteration counts or when improvements in the objective function fall below a specific threshold.
# * **Optimal Point**: The input value yielding the minimum or maximum of the objective function.
# * **Gradient**: A vector indicating both the direction and magnitude of the steepest change in a function.
# 
# <br>
# 
# <h2>A. Residual Sum of Squares (RSS)</h2><a id="section_3_A"></a>
# 
# The **Residual Sum of Squares (RSS)** serves as a metric to compute differences between real and estimated values. It's calculated using the formula:
# 
# $$
# RSS = \sum_{i=1}^{n} ( y_i - \hat{y}_i )^2
# $$
# 
# Where:
# 
# * **i**: Denotes a specific row/case.
# * **n**: Represents the total sample.
# * $y_i$ : Refers to the real value of the target.
# * $\hat{y}_i$: Refers to the estimated value of the target.
# 
# Regularization model includes some penalization into this RSS.
# 
# <br>
# 
# <h3>a. Ordinary Least Square (OLS)</h3><a id="section_3_A_a"></a>
# 
# **Ordinary Least Squares (OLS)** is a method employed to minimize the sum of the squared errors, which represent the differences between predictions and actual values.
# 
# * Usefull when the errors follows a normal distribution and there are no outliers in the data.
# * OLS is widely used in linear regression analysis to estimate the coefficients of the linear equation that best fits the observed data points. It provides a "best fit" line through the data by determining the coefficients that minimize the sum of the squared residuals.
# 
# <br>
# 
# <h2>B. Gradient Descent</h2><a id="section_3_B"></a>
# 
# **Gradient Descent** is an iterative optimization process used to find the optimal parameters of a model.
# 
# * Is an iterative process. It repeatedly updates the parameters until convergence to minimize the loss function.
# * Use the enterie sample in each iteration to update parameters.
# * Parameter are update in the oposite direction of the gradient of the loss function.
# * Search the local or global minimun.
# * While Gradient Descent might be slower than certain methods for simple linear regression problems, it offers more flexibility and is particularly useful for complex problems with non-linear relationships.
# 
# <br>
# 
# <h2>C. SGD Regressor (Stochastic Gradient Descent)</h2><a id="section_3_C"></a>
# 
# **Stochastic Gradient Descent (SGD)** is a variant of Gradient Descent used in machine learning. It seeks to minimize the loss function to update the model parameters.
# 
# * Search for the minimun to update the model parameters.
# * Use when you have large samples.
# * Unlike traditional Gradient Descent, SGD utilizes a random sample of the data in each iteration. This characteristic makes it efficient for large training samples, as it processes smaller batches of data at a time.
# 
# <br>
# 
# <h2>D. Regularization</h2><a id="section_3_D"></a>
# 
# Regularization methods are used to prevent overfitting by adding a penalty term to the loss function. This penalty term encourages the model to learn simpler patterns and helps prevent it from fitting noise in the training data. By optimizing both the original loss function and the regularization term, regularization techniques strike a balance between fitting the training data well and generalizing to unseen data, thus improving the overall performance of the model.
# 
# **Differences between Lasso and Ridge Regression**:
# 
# | | Ridge Regression (L2) | Lasso Regression (L1)|
# |:--:| :--:|:--:|
# |Penalty term| Proportional to the square of the square of the magnitude of the coefficients: $ \Theta^2 $ | Proportional to the absolute value of the coefficients $ |\Theta| $. It could drive some coefficients to 0|
# |Objetive| Reduces overfitting and model complexity| Reduces complexity and automatically selects some features (by setting their coefficients to 0. In other words, deleting it)|
# |Predictions| Generally yields better predictions| More interpretable due to feature selection|
# 
# **$\alpha$**: Hiperparameter that controls how much is going to be penalize. Higher the value, simpler the models. To select the best value, a tuning process must be done. **WARNING** in Lasso, if we pick a large value we can deleate a lot of features because their coefficients will go to 0.
# 
# <br>
# 
# <h3>a. Ridge Regression (L2)</h3><a id="section_3_D_a"></a>
# 
# When to use it:
# * High correlation among features.
# * Number of features > number of observations (overdimensional models)
# * When aiming to reduce overfitting while maintaining multicollinearity.
# 
# This model penalizes as follows:
# 
# $$
# RSS + \alpha ·  \sum_{j=1}^{p} (\hat{\beta} _j )^2
# $$
# 
# Where:
# 
# * **j**: Represent a specific feature.
# * **p**: Denotes total number of features.
# * $\beta$: Denotes the coeficient of the feature.
# * $\alpha$: Represent the hiperparameter.
# 
# <br>
# 
# <h3>b. Lasso Regression (L1)</h3><a id="section_3_D_b"></a>
# 
# This model penalizes as follow:
# 
# $$
# \frac{1}{2n}· RSS + \alpha ·  \sum_{j=1}^{p} |\hat{\beta}_j|
# $$
# 
# Where:
# 
# * **j**: Denotes a specific feature.
# * **p**: Represents the total number of features.
# * $\beta$: Represents the coeficient of the feature.
# * $\alpha$: Denotes the hiperparameter.
# * **n**: Represents the number of obervations.
# 
# <br>
# 
# <h3>c. Elastic Net</h3><a id="section_3_D_c"></a>
# 
# Combines the penalties of both Lasso (L1) and Ridge (L2) regression. It aims to overcome the limitations of each method individually by incorporating their strengths.
# 
# In Elastic Net, the loss function is augmented with two penalty terms: one that is proportional to the absolute values of the coefficients (L1 penalty), and another that is proportional to the square of the coefficients (L2 penalty). The objective of Elastic Net is to find a balance between feature selection (like Lasso) and parameter shrinkage (like Ridge).
# 
# Elastic Net is particularly useful when dealing with datasets where there are multiple correlated features, as it can select groups of correlated features together, unlike Lasso which tends to arbitrarily select only one feature from a group.
# 
# The Elastic Net hyperparameters include $\alpha$, controlling the overall strength of regularization, and the mixing parameter $\rho$, determining the balance between L1 and L2 penalties.
# 
#   <br>
# 
# 
#         
# We are going to investigate the regularization options in Python starting with **Ridge Regression**.
# First of all, we select different values for the hiperparameter $\alpha$.

# In[29]:


# Create the model with different values for alpha
ridge_parameters = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])


# <br>
# Now we fit the model and see what is the best value for $\alpha$

# In[30]:


# Fit the model
model5 = ridge_parameters.fit(df_features.sort_index(), 
                              df_target.sort_index())

# Show alpha, intercept value ,and coefficients:
print("Coefficients :", model5.alpha_)
print("Intercept :", model5.intercept_ [0].round(3))

coefficient_dict = dict(zip(df_features.columns,
                            model5.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# <br>
# 
# As we see, alpha take a optimal value of 0.1.
# 
# Now build the equation of the model:
# 
# <br>
# <div style='font-size: 14px;'>
# $$
# \hat{y} = 4.574 + 0.009 · x_1 -1.099 · x_2 - 0.187 · x_3 + 0.009 · x_4 - 1.830 · x_5 + 0.005 · x_6 - 0.003 · x_7 - 0.165 · x_8 - 0.495 · x_9 + 0.883 · x_{10} + 0.293 · x_{11} 
# $$
# </div>
# 
# If we substitute the features for our first case, we obtain the following:

# In[31]:


print("Value of all features for our first case :", df_features.sort_index().iloc[0:1].transpose().round(3)) 


# <br>
# 
# We replace the $x$ variables with the features information:
# 
# <br>
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = 4.574 + 0.009 · 7.4 -1.099 · 0.7 - 0.187 · 0 + 0.009 · 1.9 - 1.830 · 0.076 + 0.005 · 11 - 0.003 · 34 - 0.165 · 0.998 - 0.495 · 3.51 + 0.883 · 0.56 + 0.293 · 9.4 \approx 5.049
# $$
# </div>
# 
# If we compute in Python:

# In[32]:


model5_y = pd.DataFrame(model5.predict(df_features.sort_index()), 
                      index=df_features.index, 
                      columns=['Ridge Model'])

print("Quality predicted value for the frst case:", model5_y.sort_index().head(1))


# <br>
# 
# Now, we a build the model following the **Lasso Regression**. The steps to follow are the same as in Ridge Regression. So first of all, we select the optimal value of alpha and then we fix the model.

# In[33]:


# Create the model with different values for alpha
lasso_parameters = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])

# Fit the model
model6 = lasso_parameters.fit(df_features.sort_index(), 
                              df_target.sort_index().values.ravel()) # With Lasso scikit learn forces us to use an array (1-D)

# Show alpha, intercept value ,and coefficients:
print("Alpha :", model6.alpha_)
print("Intercept :", model6.intercept_.round(3))

coefficient_dict = dict(zip(df_features.columns,
                            model6.coef_))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# As we can see, here the best alpha of those we have proposed is 0.001. Now, the "density" coefficient drops to 0 which means that this characteristic doesn't affect the predictions. If we build the model's equation we obtain:
# 
# $$
# \hat{y} = 3.903 + 0.013 · x_1 -1.075 · x_2 - 0.115 · x_3 + 0.006 · x_4 - 1.248 · x_5 + 0.005 · x_6 - 0.003 · x_7 - 0.000 · x_8 - 0.364 · x_9 + 0.786 · x_{10} + 0.296 · x_{11} 
# $$
# 
# 
# If we replace the features for our first case we obtain this:

# In[34]:


print("Value of all features for our first case :", df_features.sort_index().iloc[0:1].transpose().round(3))


# In the equation:
# 
# <br>
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = 3.903 + 0.013 · 7.4 -1.075 · 0.7 - 0.115 · 0 + 0.006 · 1.9 - 1.248 · 0.076 + 0.005 · 11 - 0.003 · 34 - 0.000 · 0.998 - 0.364 · 3.510 + 0.786 · 0.560 + 0.296 · 9.4  \approx 5.061
# $$
# </div>
# 
# In Python we obtain a similar value:

# In[35]:


model6_y = pd.DataFrame(model6.predict(df_features.sort_index()), 
                        index=df_features.index, 
                        columns=['Lasso Model'])

print("Quality predicted value for the frst case:", model6_y.sort_index().round(3).head(1))


# Now, we are going to draw a scatter plot with the differences between the actual value of "quality" and the predicted values comparing three models: Multiple Linear Regression, Ridge Regression, and Lasso Regression.
# 
# No large differences are shown between the three models.

# In[36]:


plt.figure(figsize=(10, 4))


# Plot model 2
plt.subplot(1, 3, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model2_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model1_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Multiple Regression: \nPredicted vs Real Values', fontsize=10)
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


# Plot model 5
plt.subplot(1, 3, 2)

plt.scatter(df_target.sort_index(),     # Real value
                model5_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model5_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Ridge Regression: \nPredicted vs Real Values', fontsize=10)
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)

# Plot model 6
plt.subplot(1, 3, 3)

plt.scatter(df_target.sort_index(),     # Real value
                model6_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model6_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Lasso Regression: \nPredicted vs Real Values', fontsize=10)
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)

plt.show()


# <br>
# 
# <h1>4. Quantify the fit</h1><a id="section_4"></a>
# 
# <h2>A. Coefficient of determination $R^2$</h2><a id="section_4_A"></a>
# 
# **Definition**: "Proportion of target varianze that could be predict using the targets".
# 
# This coefficient is used to compare the efficiency of different models and determine which of them fits the observate data better.
# 
# The formula is as follow:
# 
# $$
# R^2 = 1 - \frac{SS_{res}}{SS_{total}}
# $$
# 
# * Residual Sum Square ($SS_{res}$): 
# 
# $$
# SS_{res} = \sum(y_i - \hat{y_i})^2
# $$
# 
# * Total Sum Square ($SS_{total}$): 
# 
# $$
# SS_{total} = \sum(y_i - \bar{y_i})^2
# $$
# 
# <br>
# 
# The **nomenclature** used in these formulas:
# 
# * $y_i$: Real values of the target.
# * $\hat{y_i}$: Target values predicted by the model.
# * $\bar{y_i}$: Mean of the real values ($y_i$).
# 
# <br>
# 
# **Interpretation**:
# 
# * $R^2$ = 1. The model is perfect. The features explain all the variability of the target. The residues are minimus.
# * $R^2$ = 0. The model doesn't explain the variability in the data. It's not more useful than using the mean of the data as a predictor of each observation.
# * $R^2$ < 0. The model performance worse than using simply the mean.
# 
# <br>
# 
# **Characteristics**:
# 
# * Although it is rare, the model could be worse than use only the mean.
# * When the prediction residuals have a zero mean, the score is identical to the Explained Variance score.
# 
# We are going to explore the models that we generate in previous steps:

# In[37]:


# Sort by index
df_target = df_target.sort_index()
model1_y = model1_y.sort_index()
model2_y = model2_y.sort_index()
model3_y = model3_y.sort_index()
model4_y = model4_y.sort_index()
model5_y = model5_y.sort_index()
model6_y = model6_y.sort_index()

# Ensure all preidctions and real observations has the same index:
assert df_target.index.equals(model1_y.index), "Indices do not match"
assert df_target.index.equals(model2_y.index), "Indices do not match"
assert df_target.index.equals(model3_y.index), "Indices do not match"
assert df_target.index.equals(model4_y.index), "Indices do not match"
assert df_target.index.equals(model5_y.index), "Indices do not match"
assert df_target.index.equals(model6_y.index), "Indices do not match"
print("Indices match successfully!")

predictions = pd.DataFrame({
    'Actual Target': np.squeeze(df_target), # Real values
    'Simple Model': np.squeeze(model1_y),
    'Multiple Model': np.squeeze(model2_y),
    'Interaction Model': np.squeeze(model3_y),
    'Polynomial Model': np.squeeze(model4_y),
    'Ridge Model': np.squeeze(model5_y),
    'Lasso Model': np.squeeze(model6_y),
})

print(predictions.head())


# Now we are going to calculate the $R^2$ coefficient:

# In[38]:


r2_scores = {}
r2_scores['Simple Model'] = r2_score(predictions['Actual Target'], predictions['Simple Model'])
r2_scores['Multiple Model'] = r2_score(predictions['Actual Target'], predictions['Multiple Model'])
r2_scores['Interaction Model'] = r2_score(predictions['Actual Target'], predictions['Interaction Model'])
r2_scores['Polynomial Model'] = r2_score(predictions['Actual Target'], predictions['Polynomial Model'])
r2_scores['Ridge Model'] = r2_score(predictions['Actual Target'], predictions['Ridge Model'])
r2_scores['Lasso Model'] = r2_score(predictions['Actual Target'], predictions['Lasso Model'])

# Print the R2 scores
for model, score in r2_scores.items():
    print(f"R2 score for {model}: {score:.4f}")


# We can see that the model that includes all the features plus the polynomial effect of "sulphite" is the one that performs better. The model that performs the worst is the simple regression model, where predictions are made only with "alcohol" information.

# <h2>B. Adjusted coefficient of determination $R^2$</h2><a id="section_4_B"></a>
# 
# **Why use it?**: When we add more predictiors to the model, $R^2$ always increase (or at least doesn´t decrease). The problem is that adjusted is not as inerpretable as $R^2$.
# 
# Formula looks like follows:
# 
# $$
# R^2 = 1 - \left( \frac{SS_{\text{res}}}{SS_{\text{total}}} \right) \cdot \left( \frac{N-1}{N-K-1} \right)
# $$
# 
# Where:
# 
# * $SS_{\text{res}}$ and $SS_{\text{total}}$ are the same as in $R^2$.
# * **N**: Number of observations in the data.
# * **K**: Number of features used in the model (excluding the intercept).
# 
# The second part of the formula is the part that adjust for Degrees of Freedom.
# 
# <br>
# <br>
# 
# 
# <h1>5. Hypothesis tests</h1><a id="section_5"></a>
# 
# Two aproximations:
# 
# * A. See if the model proposed is better than the null model.
# * B. Test if a particular coefficient is statistical different from 0.
# 
# <br>
# 
# <h2>A. Compare with the null model</h2><a id="section_5_A"></a>
# 
# The null model (baseline model) is the one that doesn't use any feature to predict the target.
# 
# The null hypothesis says that there is no relation between the features and the target.
# 
# **Null model**: Consist of a model that predict using only the mean or the median.
# 
# $$
# \hat{y_i} = \bar{y_i}:
# $$
# 
# Where:
# 
# * $\hat{y_i}$: Prediction.
# * $\bar{y_i}$: Mean of the target values in the training set.
# 
# We can follow three strategies:
# 
# <br>
# 
# 
# <h3>a. Difference in $R^2$</h3><a id="section_5_A_a"></a>
# 
# Compare the values of the coefficient of determination between the null model and an alternative model. Following this, we can know which model works better, but we cannot ensure if one works significatilly better than the other. 
# 
# <br>
# 
# <h3>b. F-test for nested models (Extra Sum of Squares Test)</h3><a id="section_5_A_b"></a>
# 
# This approach is used when the alternative model is equal to the null model plus some additional predictors.
# 
# We calculate if the explained variance in the full modell is significantly greater than in the null model relative to the increase in degrees of fredoon (number of predictors).
# 
# Formula:
# 
# $$
# F = \frac{\frac{(R^2_{\text{full}} - R^2_{\text{reduced}})}{p}}{\frac{(1 - R^2_{\text{full}})}{(n - p - 1)}} 
# $$
# 
# 
# Where:
# * $R^2_{\text{full}}$ is the R-squared of the full model (with the additional predictors).
# * $R^2_{\text{reduced}}$ is the R-squared of the reduced model (null model).
# * $p$ is the number of extra predictors added to the reduced model to get the full model.
# * $n$ is the total number of observations.
# 
# <h3>c. Information Criteria (AIC\BIC)</h3><a id="section_5_A_c"></a>
# 
# * **AIC: Akaike Information Criterion**. Focuses on selecting the model that most closely approximates the truth. **Disadvantage**: It can favor more complex model (risk of overfitting). Use when all model to compare have the same complexity or when the models are nested.
# * **BIC: Bayesian Information Criterion**. Strong penalty for complexity, often favoring simpler model especially as the sample size grows. Preferable when dealing with very large datasets.
# 
# **Characteristics**:
# 
# * Meassure the complexity of the model penalizing unnecessary complexity (too many predictors). 
# * Lower values of AIC and BIC means a better model.
# * A significant drop in AIC and BIC values when moving from the null model to the alternative model can suggest that the increase in the model complexity is justified.
# * Interesting when you are interested in model selection or balancing models.
# 
# **Interpretation**: There is no way to test if the improvements is significant. However, there is a rule of thumb. We can apply it for both AIC and BIC:
# 
# * **Diferences < 2**: No siginificant.
# * **Differences between 2 and 6**: Some evidence against the model with the higher value of the criterion. This means the model with the lower AIC\BIC is better.
# * **Differences between 6 and 10**: Strong evidence that the model with the lower criterion is better.
# * **Differences > 10**. Very strong evidence.
# 
# 
# <br>
# 
# Now we are going to implement these three approachs in Python
# <br>
# 
# ### a. Difference in $R^2$
# 
# We are going to start with the difference in $R^2$. First, we create the null value and compare with the other models we have, then we use the F-test approach and finally the Information Criteria approach.

# In[39]:


# Empty model: Always predict the mean

# Create the model
regression0 = DummyRegressor(strategy='mean')

# Fit the model
model0 = regression0.fit(df_features.sort_index(),
                         df_target.sort_index())

# Predicted value (same for all cases)
model0_y = model0.predict(df_features.sort_index())
print("Quality predicted value (same for all cases):", model0_y[0].round(3))


# In[40]:


r2_null_model = r2_score(df_target, 
                         model0_y)
print("R² of null model:", r2_null_model)


# We obtaining an $R^2$ score of 0 for the null model is expected and indicates that the model's predictions are no better than simply predicting the mean of the target variable.
# 
# <br>
# 
# Now create a table with the $R^2$ of each model:

# In[41]:


# Predictions as df
df_r2_scores = pd.DataFrame(list(r2_scores.items()), 
                            columns=['Model', 'R-squared'])

# Add the new column
null_model_row = pd.DataFrame({'Model': ['Null Model'], 
                               'R-squared': [r2_null_model]})
df_r2_scores = pd.concat([null_model_row,
                         df_r2_scores],
                         ignore_index=True)

print(df_r2_scores)


# <br>
# 
# As we can see, the all the models predict better than the null model.
# 
# <br>
# 
# Let's plot the predictions made with the null model and the multiple model (referred to as the full model in the next step). We can see the null model predict a line, because all the predictions take the same value: the mean.

# In[42]:


plt.figure(figsize=(12, 6))


# Plot model 0
plt.subplot(1, 2, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model0_y,      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model0_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Null model: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)



# Plot model 2
plt.subplot(1, 2, 2)

plt.scatter(df_target.sort_index(),     # Real value
                model2_y.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model2_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Full model (Multiple Regression): Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


plt.show()


# <br>
# 
# <h3>b. F-test for nested models (Extra Sum of Squares Test)</h3>
# 
# Now we are going to see if on specific model performs better than the null model. We are going to use model 2 (Multiple Regression Model). 
# 
# **Important**: In statsmodel (sm) we need to include a constant if we want an intercept in the model.

# In[43]:


# Creare a df for the null model. As we don´t have predictors, only a constant is present.
null_features   = sm.add_constant(pd.DataFrame(index=df_features.index)).sort_index()
full_features   = sm.add_constant(df_features).sort_index()

#reduce_model = sm.add_constant(X_reduced)
print("Null features:\n", null_features.head(1), 
      "\nMultiple Regression features:\n", full_features.tail(1))


# <br>
# 
# Use OLS to fit each model:

# In[44]:


# Fit the models using OLS
model_null = sm.OLS(df_target.sort_index(), 
                    null_features.sort_index()).fit()
model_full = sm.OLS(df_target.sort_index(), 
                    full_features.sort_index()).fit()

# Summary of the models
print("Null Model Summary:")
print(model_null.summary())
print("\nFull Model Summary:")
print(model_full.summary())


# <br>
# 
# For each model we have three tables:
# 
# * **OLS Regression Results**. We have **General model information** and **Fit statistics**.
# * **Coefficients table**. Information related with intercpt and coefficients.
# * **Residual test**.
# 
# Lest's see what information do we have. We are going to try to give a little explenation of all the fields.
# 
# <br>
# 
# **General model information**
# 
# * **Dep. Variable**: The dependent variable for the model, which in this case is 'quality'.
# * **Model**: Indicates the type of model used, here it's OLS.
# * **Method**: The method of estimation used, which is "Least Squares."
# * **Date, Time**: The date and time when the model was run.
# * **No. Observations**: Total number of observations used in the model (1599).
# * **Df Residuals**: Degrees of freedom of the residuals. There are calculated as the number of observations minus the number of parameters. Null model = 1599 - 1 (constant, intercept) = 1598. Full model = 1599 - 12 (intercept + predictors)  = 1587.
# * **Df Model**: Number of predictors in the model. The null model doesn't use predictors (0). The full model use 11 features.
# * **Covariance Type**: Specifies the type of covariance calculation method used, here it's "nonrobust".
# 
# <br>
# 
# **Fit Statistics**:
# 
# * **R-squared**: The proportion of variance in the dependent variable that is predictable from the independent variables. 0 for null model and 0.361 (as wee check before).
# * **Adj. R-squared**: Adjusted R-squared, which is adjusted for the number of predictors in the model. It is useful for comparing models with different numbers of predictors. Again, null model take 0 value and full model 0.356.
# * **F-statistic**: A measure of how significant the fit is. The higher the F-statistic, the more evidence against the null hypothesis that the model provides no better fit than a model with no predictors. Interpreta it with the associated probability.
# * **Prob (F-statistic)**: The probability of observing the data if the null hypothesis is true (null hypothesis is use the mean so is the null model). Extremely small, indicating a significant model, like we observate in the full model.
# * **Log-Likelihood**: The log of the likelihood function, which measures the goodness of fit of the model, Measure the probability of observing the data given the parameters. A higher value means the model fits better the data. However, to compare model we use to transform into AIC and BIC that takes into account the number of observations and the complexity of the model.  
# * **AIC**: Akaike Information Criterion, a measure of the relative quality of statistical models for a given set of data. Allow us to compare one model against other. Consider both the complexity of the model and the fit of the model to the dataset. Used to find the model that best explains the data with a minimun number of parameters. Lower AIC, better model. In our case, full model is better than null (lower AIC).
# * **BIC**: Bayesian Information Criterion, another criterion for model selection. Comparing with AIC, BIC includes a stronger penalty for models with more parameters. As AIC, lower BIC value indicate a better model. Useful with large datasets. Again, the full model shows a lower BIC than null model.
# 
# <br>
# 
# **Coefficients Table**
# 
# In the null model, only the constant is presented.
# 
# * **coef**: Estimated coefficients for the predictor variables. In the null model take the mean value. In the full model, if we compare with the obtained in 1.B, we see there are the samen. The sign of the coefficient is used to interpretated the relation between the current feature and the target. Negative means that more value in that characteristics supose a decrease in the target variable. A positive sign means that more value of the feature, more value of the target. 
# * **std err**: Standard error of the estimated coefficients.
# * **t**: The t-statistic, which is the coefficient divided by its standard error.
# * **P>|t|**: The p-value for a hypothesis test whose null hypothesis is that the coefficient is zero (no effect). Using an $\alpha$ = 0.05 we see that the unique variables that have an effect are: "volatile acidity", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "pH", "sulphates", and "alcohol".
# * **[0.025 0.975]** (aka Confidence intervals): The 95% confidence intervals for the coefficients.
# 
# <br>
# 
# **Residual Tests**
# * **Omnibus**: Omnibus D'Angostino's test. Provides a combined statistical test for the presence of skewness and kurtosis. When the model resides of the model are not normally distributed, this can imply that the model assumption for the linear regression are violated.This canaffect the reliability of some standard test of the coefficients, leading to incorrect conclusions. In our case, no assumption test were used so it´s none a surpise.
# * **Prob(Omnibus)**: The *p-value* associated with the Omnibus test. In models is close to 0 indicating non-normal residuals.
# * **Skew**: A measure of data symmetry- In the null model is slightly positive, indicating the tail is on the rigth side. In the full model is slightly negative, indicating the tail is on the left side.
# * **Kurtosis**: A measure of whether data are peaked or flat relative to a normal distribution. Here it indicates a leptokurtic distribution (peaked) in both model.
# * **Durbin-Watson**: A test statistic that checks for autocorrelation in the residuals from a regression analysis. In both models is close to 2, suggesting minimal autocorrelation. Low autocorrelation is good because it indicates that residuals are independent from each other, which is an assumption of OLS regression.
# * **Jarque-Bera (JB)**: A goodness-of-fit test of whether sample data have the skewness and kurtosis matching a normal distribution.
# * **Prob(JB)**: The *p-value* for the Jarque-Bera test, here indicating departure from normality in both models. This may lead to inefficiencies in the parameter estimates and standard errors, potentially affecting hypothesis tests about the coefficients.
# * **Cond. No**: Condition number, which measures the sensitivity of the function's output to its input. High values indicate potential multicollinearity or other numerical problems in the regression (ex. the scale of the input variables is not balanced). This can lead to instability in the estimation of regression coefficients, where small changes in the data might lead to large changes in the model estimates. The model might also be overly sensitive to small changes in the model or the data. We have 1 in the null model. It indicates perfect stability and no multicollinearity (since there are no features). It’s a baseline or reference point indicating no correlation or dependency issues among predictors because there aren’t any in the model.
# 
# 
# <br>
# 
# Now we compare if full model is nested within null model:

# In[45]:


f_test = model_full.compare_f_test(model_null)
print("\nF-test result:", f_test)


# **Results**:
# 
# * **F statistic value**: 81.348. This is the calculated F-statistic for the test. It represents the ratio of the model fit improvement per added predictor to the error of the larger model (including all predictors). A higher F-statistic indicates that the additional predictors in the full model provide a significant improvement in fit over the reduced model.
# * **P-value**: <0.001. Probability under the null hypothesis of observing the F-statistic, or one more extreme, given that the null hypothesis is true. Here, the null hypothesis typically states that the additional predictors in the full model do not improve the fit of the model meaningfully compared to the reduced model. A very small p-value, as in this case (which is virtually zero), strongly suggests rejecting the null hypothesis. This means that the additional predictors do significantly improve the model.
# * **Degrees of Freedom**: 11.0. This number represents the degrees of freedom associated with the numerator of the F-statistic, which is generally equal to the number of additional parameters added to the reduced model to get the full model. Here, 11 extra parameters were added to the reduced model to form the full model.
# 
# **Overall conclusion**: The features in the full model improve the prediction compared to the null model.
# 
# <br>
# 
# <h3>c. Information criteria (AIC/BIC)</h3>
# 
# Follow the same steps as in part **5.A.b** to obtain model_null and model_full. Once we have completed these steps, we will move on to examine the AIC and BIC criteria for each model.

# In[46]:


print("Null model AIC:", round(model_null.aic, 3), "Full model AIC:", round(model_full.aic, 3))
print("Null model BIC:", round(model_null.bic, 3), "Full model BIC:", round(model_full.bic, 3))


# <br>
# 
# We compute Deltas (differences) to interpretate the results:

# In[47]:


# Diferences:
delta_aic = model_null.aic - model_full.aic
delta_bic = model_null.bic - model_full.bic

print("Delta AIC:", round(delta_aic), "\nDelta BIC:", round(delta_bic))


# As we can see, the differences are large (differences >10), so we can conclude that the full model is better at making predictions than the null model.
# 
# <br>
# <br>
# 
# <h2>B. Test if a particular coefficient is statistical different from 0</h2><a id="section_5_B"></a>
# 
# This approach allows us to examine each predictor in the model separately. We can test whether a particular predictor has a statistically significant relationship with the target variable. If the coefficient is statistically different from 0, then we can conclude there is a relationship.
# 
# The Scikit-learn package does not have a direct way to compute this, but we can fit the model with Scikit-Learn and then use the statsmodel package from the stats module to explore the statistical summary.
# 
# We are going to use the model with all features that we computed in the Multiple Linear Regression section.

# In[48]:


print("Intercept :",    model2.intercept_ [0].round(3))

coefficient_dict = dict(zip(df_features.columns,
                            model2.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# <br>
# 
# Now, we'll obtain the statistical summary using statsmodels. This summary includes the standard errors, t-statistics, and *p*-values of the coefficients, as we did in the previous section.Now, we obtain the statistical summary using statsmodels. We had do this in the previous section.

# In[49]:


print(model_full.summary())


# <br>
# 
# In the table above we have a list with all the features informations as well as the intercept.
# 
# **Coefficients statistical significative** at 0.05 level:
# 
# * "Volatile acidity", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", and "alcohol".
# 
# <br>
# 
# <h1>6. Standardised coefficients</h1><a id="section_6"></a>
# 
# It is common for our variables (target) to have different scales. The units of measurement influence the refgession coefficients.
# 
# **Standardized coefficients**: Coefficients obtained when converting all the variables to standard scores ($z$-score). 
# 
# **BEFORE** running the regression. When we standarization involves making the variable have a mean 0 and a standard deviation of 1. 
# 
# **Reasons to use**:
# 
# * It is specially important when we want to compare the strenght of different predictors (features). An absolute value of $\beta$ indicates a stronger relationship with the target.
# * Coefficients of standardized variables represent the change in the dependent variable for a one standard deviation change in the predictor, making it easier to compare the effects of different variables.
# * Moreover, when we use regularization, standarization is crital because it penalize the coefficients base on their size. 
# * Accelerate convergence in algorithms that use gradient descent (like logistic regression or when using regularitation).
# 
# We will now see how to do this in Python step by step. We start with the standarization of the variables. 
# 
# **Important Note**: If we split our data into training and test sets, we may do this before the splitting.

# In[50]:


scaler_features = StandardScaler()
scaler_target = StandardScaler()

df_features_stand = scaler_features.fit_transform(df_features)
df_target_stand = scaler_target.fit_transform(df_target)

print("Features standarized: \n", df_features_stand[0][0:])
print("Target standarize: \n", df_target_stand[0][0:3])


# <br>
# 
# To use StandarScaler() it is necessary transform our data into an array. We need to recover the index and the columns names before made the regression

# In[51]:


# We made for features and target
df_target_stand = pd.DataFrame(df_target_stand, 
                               index=df_target.index, 
                               columns=df_target.columns).sort_index()

df_features_stand = pd.DataFrame(df_features_stand, 
                                       index=df_features.index, 
                                       columns=df_features.columns).sort_index()


# Print the standardized features with the DataFrame structure
print(df_features_stand.head(3))


# <br>
# 
# Create and fit the model:

# In[52]:


# Create MLRM and fit it
regression_stand = LinearRegression()

# Fit the model
model7 = regression_stand.fit(df_features_stand.sort_index(),
                              df_target_stand.sort_index())

# Show the equation values:
print("Intercept :",    model7.intercept_ [0].round(3))

coefficient_dict = dict(zip(df_features_stand.columns,
                            model7.coef_[0]))
print("Coefficients :")
for feature, coef in coefficient_dict.items():
    print(f"{feature}: {coef:.3f}")


# <br>
# 
# Now that we have all standarized, we are going to interpretate the intercept and coefficients:
# 
# * **Intercept** (-0.0): Since the model's intercept is effectively zero, it suggests that the model predicts a response value of zero when all predictors are at their mean values.
# * **Fixed Acidity** (0.054): This positive coefficient indicates that increasing fixed acidity by one standard deviation, while holding other variables constant, increases the response variable by 0.054 standard deviations. It shows a moderate positive effect on the response variable.
# * **Volatile Acidity** (-0.240): The substantial negative coefficient implies that increasing volatile acidity by one standard deviation results in a decrease in the response variable by 0.240 standard deviations. This indicates a significant adverse effect.
# * **Citric Acid** (-0.044): A small negative coefficient suggests that an increase in citric acid slightly decreases the response variable, though the effect is relatively minor.
# * **Residual Sugar** (0.029): A modest positive coefficient implies that higher residual sugar levels slightly increase the response variable.
# * **Chlorides** (-0.109): A negative effect where an increase in chlorides results in a decrease in the response variable, indicating a detrimental impact.
# * **Free Sulfur Dioxide** (0.056): Indicates a positive relationship, with increases in free sulfur dioxide associated with slight increases in the response variable.
# * **Total Sulfur Dioxide** (-0.133): Reflects a negative effect, with an increase in total sulfur dioxide resulting in a decrease in the response variable.
# * **Density** (-0.042): A small negative coefficient indicates that higher density slightly reduces the response variable.
# * **pH** (-0.079): This negative coefficient suggests that higher pH values are associated with decreases in the response variable.
# * **Sulphates** (0.192): A robust positive effect, suggesting that increases in sulphates lead to significant increases in the response variable.
# * **Alcohol** (0.364): The most significant positive coefficient, indicating that an increase in alcohol content substantially increases the response variable.
# 
# 

# 
# Now build the equation of the model:
# 
# <br>
# 
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = 0 + 0.054 · x_1 - 0.240 · x_2 - 0.044 · x_3 + 0.029 · x_4  - 0.109 · x_5 + 0.056 · x_6 - 0.133 · x_7 - 0.042 · x_8 - 0.079 · x_9 + 0.192· x_{10} + 0.364 · x_{11} 
# $$
# </div>
# 
# If we replace the features for our first case, we obtain the following (using the standardized features):"

# In[53]:


print("Value of all features for our first case :", df_features_stand.sort_index().iloc[0:1].transpose().round(3)) 


# <br>
# 
# The equation for our first case is like:
# 
# <br>
# <div style='font-size: 14px;'>
# 
# $$
# \hat{y} = 0 + 0.054 · (-0.528) - 0.240 · 0.962 - 0.044 · (-1.391) + 0.029 · (-0.453)  - 0.109 · (-0.244) + 0.056 · (-0.466) - 0.133 · (-0.379) - 0.042 · 0.558 - 0.079 · 1.289 + 0.192 · (-0.579) + 0.364 · (-0.960) \approx 0.746
# $$
# </div>

# In[54]:


model7_y_stand = pd.DataFrame(model7.predict(df_features_stand), 
                        index=df_features_stand.index, 
                        columns=['Standarized model'])

print("Quality predicted value for the first case:", model7_y_stand.sort_index().round(3).head(1))


# <br>
# 
# Standarize the variables has several advantages (scale independence, improve algorithm performance, etc). However, the interpretation of the results is complicated. For this reason we need to get our predicionts back to the original scale. 

# In[55]:


df_target_des_stand = scaler_target.inverse_transform(model7_y_stand)

df_target_des_stand = pd.DataFrame(df_target_des_stand, 
                                     index=df_target_stand.index, 
                                     columns=['Original Scale Prediction'])

print("Quality predicted value for the first case (Multiple Linear Regression):", model2_y.sort_index().round(3).head(1))
print("Quality predicted value for the first case (After de-standardizing ):", df_target_des_stand.sort_index().round(3).head(1))


# <br>
# 
# As we can see, when we return to the original scale, we obtaing the same results that using th Multiple Linear Regression.
# 
# <br>
# 
# We will plot the predictions made with the Multiple Linear Regression model (model 2) with this with the standarized values (model 7). 

# In[56]:


plt.figure(figsize=(12, 6))


# Plot model 2
plt.subplot(1, 2, 1)

plt.scatter(df_target.sort_index(),     # Real value
            model2_y.sort_index(),      # Predicted
            color='lightskyblue',
            edgecolor = "black",
            label='Predicted vs Real',
            alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target), np.max(model2_y))
plt.plot([0, max_val], [0, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(0,10+1, 1))
plt.yticks(np.arange(0,10+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Full model no standarized: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)



# Plot model 7
plt.subplot(1, 2, 2)

plt.scatter(df_target_stand.sort_index(),     # Real value
                model7_y_stand.sort_index(),      # Predicted
                color='lightskyblue',
                edgecolor = "black",
                label='Predicted vs Real',
                alpha = 0.3)

# Plot a diagonal line representing perfect predictions and customize
max_val = max(np.max(df_target_stand), np.max(model7_y_stand))
plt.plot([-3, max_val], [-3, max_val], 
         color='red', 
         linestyle='--', 
         label='Perfect Prediction')

plt.xticks(np.arange(-3,3+1, 1))
plt.yticks(np.arange(-3,3+1, 1))
plt.xlabel('Real "Quality"')
plt.ylabel('Predicted "Quality"')
plt.title('Full model standarized: Predicted vs Real Values')
plt.legend()
plt.grid(color = "gray",
         alpha = 0.3)


plt.show()

