import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor 


def pipeline(x_train, x_test, y_train, y_test, scaler, imputer, oversampling, VIF):
    """
    Args:
    - x_train, x_test: numpy arrays without the class
    - scaler: 'standard'
    - imputer: 'mean', 'median'
    - oversampling: 'smote', 'random'
    - VIF: 'yes', 'no'

    """

    if scaler == 'standard':
        x_train, x_test = StandardScalerFunction(x_train, x_test)
    
    if imputer =='mean':
        x_train, x_test = MeanImputerFunction(x_train, x_test)
    elif imputer =='median':
        x_train, x_test = MedianImputerFunction(x_train, x_test)

    if oversampling =='smote':
        x_train, y_train = SMOTEFunction(x_train, y_train)
    elif oversampling =='random':
        x_train, y_train = RandomOverSamplerFunction(x_train, y_train)
    elif oversampling == 'mixed':
        x_train, y_train = mixed(x_train, y_train)

    if VIF == 'yes':
        x_train, x_test = VIF_function(x_train, x_test)

    return x_train, x_test, y_train, y_test

    



# ---------------------------------------------------features ----------
def VIF_function(x_train, x_test, tolerance = 5):
    """x_train and x_test numpy arrays
    """

    x_train_df = pd.DataFrame(x_train)

    # VIF dataframe 
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_train_df.columns 
    
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(x_train_df.values, i) 
                            for i in range(len(x_train_df.columns))] 
    
    print(vif_data)

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    print(f'{vif_data["VIF"].round(2)}')

    x_test_df = pd.DataFrame(x_test)

    high_vif_features = vif_data[vif_data["VIF"] > tolerance]["feature"]
    x_train_vif = x_train_df.drop(columns=high_vif_features)
    x_test_vif = x_test_df.drop(columns=high_vif_features)

    x_train_vif = x_train_vif.to_numpy()
    x_test_vif = x_test_vif.to_numpy()

    return x_train_vif, x_test_vif

def getVIF(x_train):
    """x_train numpy array or pd dataframe
    """

    if not isinstance(x_train,pd.core.frame.DataFrame):
        x_train_df = pd.DataFrame(x_train)
    else:
        x_train_df = x_train

    # VIF dataframe 
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_train_df.columns 
    
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(x_train_df.values, i) 
                            for i in range(len(x_train_df.columns))] 
    

    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Save to text file
    vif_data_rounded = vif_data.round(2)
    vif_data_rounded.to_csv('vif_data', sep='\t', index=False)
    print(f'{vif_data["VIF"].round(2)}')

# ------------------------------------------------------ oversample ----------
def SMOTEFunction(x_train, y_train ):
    """
    x_train and x_test without the class
    """
    

    sm = SMOTE(random_state=0)

    x_train, y_train = sm.fit_resample(x_train, y_train)
    return x_train, y_train

def RandomOverSamplerFunction(x_train, y_train):
    ros = RandomOverSampler(random_state=0)
    x_train, y_train = ros.fit_resample(x_train, y_train)
    return x_train, y_train

def mixed(x_train, y_train):
    ros = RandomOverSampler(sampling_strategy=0.4, random_state=0)
    x_train, y_train = ros.fit_resample(x_train, y_train)
    sm = SMOTE(random_state=0)
    x_train, y_train = sm.fit_resample(x_train, y_train)
    return x_train, y_train
#----------------------------------------------------- impute -----------
def MedianImputerFunction(x_train, x_test):
    """
    x_train and x_test without the class
    """
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(x_train)
    x_train = imp_mean.transform(x_train)
    x_test = imp_mean.transform(x_test)

    return x_train, x_test

def MedianImputerFunction_OneSet(x):
    """
    x_train and x_test without the class
    """
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(x)
    x = imp_mean.transform(x)

    return x

def MeanImputerFunction(x_train, x_test):
    """
    x_train and x_test without the class
    """
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(x_train)
    x_train = imp_mean.transform(x_train)
    x_test = imp_mean.transform(x_test)

    return x_train, x_test

# ----------------------------------------- SCALE -----------------------
def StandardScalerFunction(x_train, x_test):
    """
    x_train and x_test without the class
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test

def StandardScalerFunction_OneSet(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x