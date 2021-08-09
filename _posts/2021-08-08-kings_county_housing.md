# Calculating the Most Lucrative Renovations Based on Kings County Housing Dataset


![describe_data](https://user-images.githubusercontent.com/71570329/128657233-cd2279ea-54d3-4d8c-a947-e1bf6e36bc03.png)

My second data science project with Flatiron School is an analysis of housing prices from the  [Kings County House Sales Dataset](https://www.kaggle.com/harlfoxem/housesalesprediction). 

Feel free to fork my [project repo](https://github.com/clairesarraille/ph2finproj) if you'd like to follow along and examine the code more closely.

## Business Case

The goal for this project was to model independent variables of homes in order to predict their prices. The resulting equation from running a linear regression model includes coefficients which tell us how much each independent variable is multiplied to arrive at a given predicted price.

Regressing our independent variables on our dependent variables to create a model involves several assumption which must be taken into consideration, including linearity and homoscedasticity. Additionally we not only need to take the R-squared value into consideration when we evaluate our model - we must also examine the p-values of each of our coefficients. The p-values represent the probability that, given a hypothetical situation wherein none of our independent variables impact the dependent variable at all, that we would observe an impact nonetheless due to random chance/noise. If our p-value is very low, it tells us that the impact of a given independent variable is very unlikely due to random chance or noise in the sample.

- What is **target**
>'price'
- What are **predictors**
>'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built'
- Distribution of Data - See Distribution Section

## Data Quality 
Quality of data was quite good; the only missing values were waterfront, which I converted to 0.

```python
df.isnull().sum()

>>>
id                0
price             0
bedrooms          0
bathrooms         0
sqft_living       0
sqft_lot          0
floors            0
waterfront     2376
condition         0
grade             0
yr_built          0

# Note: waterfront is our only categorical value.
# We don't need to use dummy coding or any other coding system because it's already dichotomous (1 or 0) 
df['waterfront'] = df['waterfront'].fillna(0)
df['waterfront'] = df["waterfront"].astype(int)
df['waterfront'].unique()

>>> array([0, 1])
```

## Data Exploration
One of the most interesting initial distributions I visualized was a probability density plot of the sale price of homes split into waterfront and non-waterfront properties. This showed me that the distributions of those two segments are rather distinct.

```python
# The smooth line in the visualization below is an estimate of the distributions of waterfront and non-waterfront house prices
# The parameter bandwidth rules the smoothness of the underlying distribution

# The problem with a Probability Density Function plot, is that all "point probabilities" are 0
# We must calculate the area under the curve for an interval to get the actual probability for an interval of house prices.
# Thus, it's not intuitive or easy to "read" the y-axis to get probabilities for continuous variables using a PDF like below.

plt.figure(figsize = (12,8))
sns.distplot(water_df.price_millions,kde=True)
sns.distplot(no_water_df.price_millions,kde=True)
plt.title('Comparing House Prices: Waterfront vs. Non-Waterfront')
plt.show()
```

![kde](https://user-images.githubusercontent.com/71570329/128659250-f3c87dba-7f8c-4cab-a276-cf36adb786c6.png)


## Normalization of Data
- The z-score normalization method is the most common
- If standard deviation is a yardstick, then a z-score is the measurement expressed in terms of that yardstick
- For example, if we find the z-score for a given x-value's distance from x-bar  divided by the standard deviation...
- We have convereted the distance of x from x-bar to "standard deviation units"
- If s = 30, then we want to see how many s's (quantities of 30) that x is from x-bar
- Basically, a z-score expresses each value of x as a standard unit away from the mean (x-bar)

``` python
def z_score_norm(my_column):
    return (my_column - my_column.mean())/my_column.std()
```


## Addressing Multicollinearity
- This a phenemonen where two variables we are using as predictors are correlated with each other
- This violates one of the assumptions of performing linear regression - that all independent variables are independent from one another
- If we left all features in the model without addressing multicollinearity, it would become very hard for the model to estimate the relationship between independent variables and the dependent variable, because rather than change independently, the features would change in pairs or groups.


```python
# We want to write a simple loop to use statsmodel's variance_inflation_factort method on each array (row) of our dataframe, for each column 
# That's why we use X.shape[1] as the range in the loop below, since the second term in the output of the .shape method is the number of columns
X = sm.add_constant(df_norm.drop(['price','price_millions'], axis=1))
pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
              
>>>
const          1.000000
bedrooms       1.628887
bathrooms      3.186926
sqft_living    4.152240
sqft_lot       1.047462
floors         1.573844
waterfront     1.021879
condition      1.182857
grade          2.888261
yr_built       1.734538
```

- I tried removing sqft_living because it has the highest coefficient of multicollinearity.

## Modeling

### Using all data
- I tried using all of the data, without sub-setting at all
- The results of running the linear regression model for all data was R-squared value of .646 - meaning the model descibed about 65% of the variation in the data.

![p values](https://user-images.githubusercontent.com/71570329/128660126-d60b84a1-4911-4cd5-9aae-8f107b5efb66.png)

- Unfortunately, when I tested this model for linearity and homoscedasticity, there was a problem

Plotting price in millions versus predicted price in millions resulted in this non-linear shape:
![all_cols_price](https://user-images.githubusercontent.com/71570329/128659946-5a52c80d-468e-4ba5-aa1b-dd6fd70ef002.png)

A plot of the amount of error we got from our OLS Linear Regression (Predicted Price in Millions vs. Residuals) - resulted in this distinct trumpet shape. This means there is high heteroscedasticity in the model. We want homoscedasticity -- a fairly even distribution of error.

![residuals](https://user-images.githubusercontent.com/71570329/128660054-df46ce82-dbbd-4a45-a97e-89eb913073b9.png)


### Sub-setting by Waterfront
- At this point, I re-visited my business case. I hypothesized that creating a sub-set of just those homes with a waterfront view might conform to the assumptions of linear regression. Additionally, recommending renovations to customers with waterfront view homes could maximize our fictional real estate company's profits, such as in the case of a modest home with a view but not much square footage or updated amenities.

- Sub-setting by waterfront homes resulted in a linear regression model with a decent R-squared value and a couple of features with decent p-values:

![waterfront](https://user-images.githubusercontent.com/71570329/128660371-a33113e8-967e-4a3e-8fc8-0d096aafcb2f.png)

- Additionally, plotting price in millions versus predicted price resluted in a much more linear shape:

![residuals 3](https://user-images.githubusercontent.com/71570329/128660442-c7ddf79c-7929-49fd-97e1-5f4a197f6237.png)

- And the plots of the residuals show fairly good homoscedasticity:

![residuals 1](https://user-images.githubusercontent.com/71570329/128660461-68049ee9-0758-4ee2-a326-197c40f1c7f6.png)
![residuals 2](https://user-images.githubusercontent.com/71570329/128660467-1085b596-93ae-42af-bd23-2ceb116fdffa.png)


### Interpreting Coefficients

- Now that I found a model that conformed to necessary assumptions for linear regression and resulted with a couple of coefficients with good p-values, I needed to do the reverse transformation on the coefficient values to be able to interpret their values.

``` python
waterfront_coef_list

>>>
bedrooms       0.0094
bathrooms      0.0601
sqft_living    0.4775
sqft_lot      -0.0506
floors        -0.0117
waterfront     0.0779
condition      0.0580
grade          0.1040
yr_built      -0.0737
```

- As you'll remember, these values are based on data that are in standard deviated units, converted so using the z-score normalization technique.
- To transform them to dollar amounts, I used this bit of code, and the result is in millions of dollars.

``` python
trans_coef_list = (waterfront_coef_list + waterfront_coef_list.mean()) * waterfront_coef_list.std() 

>>>
bedrooms       0.013335
bathrooms      0.021608
sqft_living    0.089717
sqft_lot       0.003545
floors         0.009892
waterfront     0.024513
condition      0.021265
grade          0.028771
yr_built      -0.000225
```

# Conclusion
- For every increase in grade, the house will sell for approximately $28,771 more.
- For every square-foot increase, the house will sell for approximately $89,717 more.
- Recommendations to home-owners in Kings County who have Waterfront properties:
    - Increase square-footage as much as possible
    - Make improvements that put home that give home a better grade according to King County's Grading System for Buildings such as:
        - Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage.
        - Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options.
        - Large amount of highest quality cabinet work, wood trim, marble, entry ways etc
