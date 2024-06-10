# Modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------------------------------------#
# Data reading functions
def ReadEnergetic(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Burst_energetics.txt" 
    file in the folder "data" and returns the data in the form of a pandas DataFrame. 
    
    The columns name are consistent with the explained labels in the header section of the text file.
    '''
    col_names = ['ID', 'z', 'S', 'e_S', 'E_S', 'Tp1024', 'f_Tp124', 'Fp1024', 
                 'e_Fp1024', 'E_Fp1024', 'Tp64', 'Fp64', 'e_Fp64', 'E_Fp64', 
                 'Tp64r', 'Fp64r', 'e_Fp64r', 'E_Fp64r', 'Eiso', 'e_Eiso', 
                 'Liso', 'e_Liso', 'Flim', 'zmax']

    # I've added this so that it works for the python notebook.
    if dir_path != '':
        dir_path += '/'

    file_path = dir_path + 'data/Burst_energetics.txt'

    energetic_data = pd.read_csv(file_path, sep=r'\s+', 
                                 skiprows=47, names=col_names)
    
    return energetic_data

def ReadSpectral(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Burst_energetics.txt" 
    file in the folder "data" and returns the data in the form of a pandas DataFrame. 
    
    The columns name are consistent with the explained labels in the header section of the file.
    '''
    
    col_names2 = ['ID', 'SType', 'tstart', 'DeltaT', 'SMod', 'f_SMod', 'alpha', 
                 'e_alpha', 'E_alpha', 'f_beta', 'beta', 'e_beta', 'E_beta', 
                 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P', 
                 'P', 'Com']

    if dir_path != '':
        dir_path += '/'

    file_path2 = dir_path + 'data/Spectral_parameters.txt'

    spectral_data = pd.read_csv(file_path2, sep=r'\s+', 
                                skiprows=57, names=col_names2)
    
    '''
    The dataset has to be rearrange because of the empty spaces in the txt file.
    To fix that, the function takes the f_beta columns (the flag on beta) that
    could be 0 or 1. 
    If f_beta is 0, beta is the best fit value in the Band Model, so
    there are values of beta and the corresponding errors (upper and lower), and 
    the row of the DataFrame is ok without modifications.
    If f_beta is 1, the value of beta is provided by (beta_min - beta) -> (see section 4.2.2 of the paper)
    so we don't have the errors on beta and we have to transpose the values in the row.
    If f_beta isn't 0 or 1, we are in the CPL Model and the "read_csv" function had skipped the 
    empty spaces because we don't have values of f_beta, beta and correlated errors, so we transpose 
    the values in that row.
    An exeption occur with a point in the dataset corresponding to SMod == PL, so it is also fixed 
    that corresponding row. 
    '''
    
    # Defining the array of f_beta from the dataframe
    arr = spectral_data['f_beta'].values
    
    # Initializing a loop taking the values in the array and the corresponding indexes
    for index, a in enumerate(arr):
        if a == 0: 
            l1 = []
            l2 = []
            l3 = []

        elif a == 1:
            # Defining the lists to transpose the values in the rows
            l1 = ['Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P', 'P', 'Com']
            l2 = ['e_beta', 'E_beta', 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P']
            l3 = ['e_beta', 'E_beta']    

        # Fixing the exeption of SMod == PL
        elif spectral_data.at[index, 'SMod'] == 'PL':
            l1 = ['F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P', 'P', 'Com']
            l2 = ['f_beta', 'beta', 'e_beta', 'E_beta', 'Ep', 'e_Ep', 'E_Ep', 'F']
            l3 = ['f_beta', 'beta', 'e_beta', 'E_beta', 'Ep', 'e_Ep', 'E_Ep']

        else:
            l1 = ['Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P', 'P', 'Com']
            l2 = ['f_beta', 'beta', 'e_beta', 'E_beta', 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2']
            l3 = ['f_beta', 'beta', 'e_beta', 'E_beta']

        # Here I use reversed to not rewrite the values after they changes
        for f1, f2 in zip(list(reversed(l1)), list(reversed(l2))):
            # Transposing the values
            spectral_data.at[index, f1] = spectral_data.at[index, f2]
        for f3 in l3:
            # Filling the empty spaces with Nan values
            spectral_data.at[index, f3] = np.nan
    
    return spectral_data

def DataReading(self, dir_path, Energetic=True, Spectral=True):
    '''
    This function uses together the previos functions to read the data from the txt files in 
    the directory path provided by the user. One can change the input boolean values of 
    "Energetic" and "Spectral" if he want to read only one of the txt file.
    After reading the data the function saves it as a pandas DataFrame in the class attributes.
    '''
    if Energetic:
        self.energetic_data = ReadEnergetic(dir_path)
    if Spectral:
        self.spectral_data = ReadSpectral(dir_path)

# Just a simple setter
def SetModel(self, model):
    self.model = model
#-------------------------------------------------------------------------------------------------#
def CorrMatrix(self, DataFrame, filter=None):
    '''
    This function is used to draw a correlation matrix from the features of the dataset.
    One can chose to add a filter list of the features if he don't want to work with all of that.
    '''
    if filter == None:
        matrix = DataFrame.corr()
    else:
        matrix = DataFrame[filter].corr()
    sns.heatmap(matrix, cmap="Greens", annot=True)
    
#---------------------------------------------------------------------------------------------------#
'''
This function takes the data from a pandas dataframe and return the values of X and y 
to use in the model. This works filtering z for y and then dropping z from the dataframe 
to have all the features without z for X.
'''
def GetData(self, df):
    y_temp = df[['z']].to_numpy()
    df.drop(['z'], axis=1)
    X = df.to_numpy()
    
    y = y_temp.reshape(-1)      # reshape to a 1-D array
    
    # Saving in the class
    self.X = X
    self.y = y
    
    return X, y    

def Run(self, df, train_size = None, test_size = None, random_state = None, criterion='squared_error', 
        max_features = 1.0 ,n_estimators = 100, max_depth = None, min_samples_leaf = 1):
    '''
    This function builds the Random Forest Regressor model and saves the 
    values of X and y for the training and testing parts, the predicted y on the X_test 
    and the model itself in the class.
    '''
    # Taking the X and Y from the DataFrame
    X, y = self.GetData(df)
    
    # Splitting X and y in the training and testing X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, 
                                                        test_size=test_size, random_state=random_state)
    
    # Initializing the model
    rnd_forest = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
    # Fitting the model with the training data
    rnd_forest.fit(X_train, y_train.ravel())
    
    # Predicting the redshift from the testing X
    predictions = rnd_forest.predict(X_test)
    
    # Saving values in the class
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.y_pred = predictions
    self.forest = rnd_forest

#------------------------------------------------------------------------------------------# 
def GridSearch(self, df, grid, scoring = None,
               cv = None, train_size = None, test_size = None):         # work in progress...
    '''
    Incomplete...
    For now I'm just testing if it works, next I provide an implementation of this 
    function so that one can put in input the values he wants to use the function 
    with more flexibility.
    '''
    
    # Taking the X and Y from the DataFrame
    X, y = self.GetData(df)
    
    # Splitting X and y in the training and testing X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)
    
    # Creating a pipeline of actions
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('rf', RandomForestRegressor()) 
    ])
    
    rf_cv = GridSearchCV(pipe, param_grid=grid, scoring=scoring, cv=cv, return_train_score=True)
    rf_cv.fit(X_train, y_train)
    
    # Saving values in the class
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.Grid_search_class = rf_cv