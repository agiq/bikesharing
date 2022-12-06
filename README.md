# Bike Sharing Analysis
A bike-sharing system is a service in which it made bikes available for shared use to individuals on a short-term basis for a price or free. The company is struggling to cope with the current market conditions. You feel that your business is being crippled by the lockdown, and that it will take a long time to recover once the lockdown is over. You feel you need to do something to prepare yourself for when things get back to normal. The consulting company wants to understand the factors affecting the demand for these shared bikes in the American market.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
The objective is to find which variable(s) is/are significant in predicting bike rental 'cnt'.  First, perform exploratory data analysis. 

- There are 730 rows and 16 columns in this data set.
- The data set has 8 categorical data and 8 numerical data after data type conversion.
- Fortunately, there is 0 missing data and 0 duplicate values.
- The Target variable is 'cnt'.
- Binary columns (with 0 & 1) are: yr, holiday, and working day
- Columns that can turn into dummies variable is: weathsit
- Columns that can turn into categorical are: 'season','mnth','yr','holiday','weekday','workingday','weathersit','day'
 
 Next, perform data visualization using boxplot on categorical data and use heatmap on numerical data.  

Analysis of categorical data:
- Falls has the highest rental. Summer is the next highest. Both seasons are when kids are out of school.  People spend time with them families and friends.
- Month: More bike rentals occur between May and September
- Year: 2019 bike rental is higher than 2018 suggesting the trend it up.
- Holiday: non-holiday (0) has a wider range compared to holiday
- Weekday: Wednesday and Thursday appear to have more bike rental
- Workingday: inconclusive – they are about the same
- Weathersit: Bike rental is high when it is Clear, Few clouds, Partly cloudy, Partly cloudy
- Day: It seems that bike rental occurs more in the middle of the month

Analysis of numerical data showing correlation.
- registered is 0.95
- casual is 0.67
- atemp, instant, temp are all 0.63

Next, do a groupby of each numeric variable with 'cnt' and plot them.  These chart provide a secondary confirmation that registered and casual are highly correlated. They show linear uptrend lines.

As part of data visualization, conduct a stacked histogram plot to show which features in each categorical data are highest. This is to provide another confirmation of our observeration above.

For model training, validating, and evaluating, we perform simple linear regression , multilinear regression and RFE training.  Once completed, we evaluate the trained model. The objective is to show manual add/remove variable, have the computer automagically select variables, and perfom a hybrid approach.


## Conclusions
- Simple Linear Regression (SLR): temp, atemp, casual and registerd displayed a linear red line upward.  The residual analysis depicted via histogram is normally distributed.  There is no identifiable pattern found.
- Multi Linear Regression (MLR): first add one variable, traing the model and reiterate. In this process, observe the R-squared and P-value. The second method is to add all variables. Then, remove the highest P-value one at a time. Utilize the corresponding high VIF value to identify the next highest variable to drop.  Only drop one at a time.
- Use Recursive Feature Elimination (RFE) to automate which features to include in training.
- Manual selection of variables for training is a tedious job. The automated feature selection is nice, but the computer is not a subject matter expert in your industry. Therefore, a hybrid approach is recommended.

For each of the training method(SLR, MLR, and RFE), plot the error terms, and get r2_score. Also, plot y_test against y_test_pred on a scatter plot.

## Technologies Used
Use the latest version of the following libraries:
- pandas - data structures and operations for manipulating numerical tables 
- numpy - scientific computing in Python
- matplotlib.pyplot - graph and chart
- seaborn - graph and chart 

Below are machine learning libraries
- sklearn 
- sklearn.model_selection
- sklearn.preprocessing  MinMaxScaler
- sklearn.metrics, r2_score
- sklearn.metrics, mean_squared_error
- statsmodels.api
 statsmodels.stats.outliers_influence, variance_inflation_factor

Use datetime to work with date and time
- datetime

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


## Contact
Created by [@agiq] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->