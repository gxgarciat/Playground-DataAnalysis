from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np


# Gathering data


boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

#data.head()

features = data.drop(['INDUS','AGE'],axis=1)

log_prices = np.log(boston_dataset.target)
#log_prices.shape
#feaures.shape

target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

# Estimation obtained from ZILLOW

ZILLOW_MEDIAN_PRICE = 583.3

# Calculating "inflation factor"
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats = features.mean().values.reshape(1,11) # template to make predictions

regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)

# MSE & RMSE
MSE = mean_squared_error(target,fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                    students_per_classrooms,
                    next_to_river = False,
                    high_confidence=True):  # These will make these arguments to be optionals
    
    # Configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classrooms
    
    # Flow control for the river
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Making prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    # Including cost range
    
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
        
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
        
        
    return log_estimate, upper_bound, lower_bound, interval

def get_dollar_estimate(rm, ptratio, chas=False, large_range = True):
    
    # Adding documentation to this function by using docstring
    """ Estimate the price of a property in Boston.
    
    Keyword arguments:
    rm -- number of rooms in the property
    ptratio -- numer of students per teacher in the classroom for the school in the are
    chas -- True if the property is next to the river. False otherwise.
    large_range -- True for a 95% prediction interval, False for a 68% interval.
    
    """
    
    # Avoiding wrong input data
    if rm < 1 or ptratio < 1:
        print('These values are unrealistic. Try again.')
        return
    
    # Calling function by using keywords
    log_est, upper, lower, conf = get_log_estimate(rm,
                                                   students_per_classrooms = ptratio,
                                                   next_to_river = chas,
                                                   high_confidence = large_range)

    # Updating to today's dollar
    dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
    dollar_low = np.e**lower * 1000 * SCALE_FACTOR

    # Rounding estimations
    rounded_est = np.around(dollar_est,-3)
    rounded_hi = np.around(dollar_hi,-3)
    rounded_low = np.around(dollar_low,-3)

    print(f'The estimated property value is: {rounded_est}')
    print(f'At {conf}% the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')
