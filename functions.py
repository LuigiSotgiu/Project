# Modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------------------------------------#
# Data crafting functions
def ReadEnergeticI(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Burst_energetics_I.txt" 
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

    file_path = dir_path + 'data/Burst_energetics_I.txt'

    energetic_data_I = pd.read_csv(file_path, sep=r'\s+', 
                                 skiprows=47, names=col_names)
    
    return energetic_data_I

def ReadEnergeticII(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Burst_energetics_II.txt" 
    file in the folder "data" and returns the data in the form of a pandas DataFrame. 
    
    The columns names are consistent with the explained in the header section of the text file.
    labels but some changes were done to be consistent with the energetic_data_I columns names too.
    '''
    col_names = ['ID', 'z', 'S', 'e_S', 'E_S', 'Tp1024', 'Fp1024', 
                 'e_Fp1024', 'E_Fp1024', 'Tp64', 'Fp64', 'e_Fp64', 'E_Fp64', 
                 'Tp64r', 'Fp64r', 'e_Fp64r', 'E_Fp64r', 'Eiso', 'e_Eiso', 
                 'E_Eiso', 'Liso', 'e_Liso', 'E_Liso']

    # I've added this so that it works for the python notebook.
    if dir_path != '':
        dir_path += '/'

    file_path = dir_path + 'data/Burst_energetics_II.txt'

    energetic_data_II = pd.read_csv(file_path, sep=r'\s+', 
                                    skiprows=40, names=col_names)
    
    return energetic_data_II

def ReadSpectralI(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Spectral_parameters_I.txt" 
    file in the folder "data" and returns the data in the form of a pandas DataFrame. 
    
    The columns name are consistent with the explained labels in the header section of the file.
    '''
    
    col_names2 = ['ID', 'SType', 'tstart', 'DeltaT', 'SMod', 'f_SMod', 'alpha', 
                 'e_alpha', 'E_alpha', 'f_beta', 'beta', 'e_beta', 'E_beta', 
                 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'f_P', 
                 'P', 'Com']

    if dir_path != '':
        dir_path += '/'

    file_path2 = dir_path + 'data/Spectral_parameters_I.txt'

    spectral_data_I = pd.read_csv(file_path2, sep=r'\s+', 
                                skiprows=57, names=col_names2)
    
    '''
    The dataset has to be rearrange because of the empty spaces in the txt file.
    To fix that, the function takes the f_beta columns (the flag on beta) that
    could be 0 or 1. 
    
    If f_beta is 0, beta is the best fit value in the Band Model, so
    there are values of beta and the corresponding errors (upper and lower), and 
    the row of the DataFrame is ok without modifications.
    
    If f_beta is 1, the value of beta is provided by (beta_min - beta) -> (see section 4.2.2 of the paper I)
    so we don't have the errors on beta and we have to transpose the values in the row.
    
    If f_beta isn't 0 or 1, we are in the CPL Model and the "read_csv" function skipped the 
    empty spaces because we don't have values of f_beta, beta and correlated errors, so we transpose 
    the values in that row.
    
    An exeption occur with a point in the dataset corresponding to SMod == PL, so it is also fixed 
    that corresponding row. 
    '''
    
    # Defining the array of f_beta from the dataframe
    arr = spectral_data_I['f_beta'].to_numpy()
    
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
        elif spectral_data_I.at[index, 'SMod'] == 'PL':
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
            spectral_data_I.at[index, f1] = spectral_data_I.at[index, f2]
        for f3 in l3:
            # Filling the empty spaces with Nan values
            spectral_data_I.at[index, f3] = np.nan
    
    return spectral_data_I

def ReadSpectralII(dir_path):
    '''
    This function takes in input the directory path, reads the data from the "Spectral_parameters_II.txt" 
    file in the folder "data" and returns the data in the form of a pandas DataFrame. 
    
    The columns name are consistent with the explained labels in the header section of the file, but 
    some changes were done to be consistent with the spectral_data_I columns names too.
    '''
    
    col_names = ['ID', 'SType', 'tstart', 'DeltaT', 'SMod', 'alpha', 
                 'e_alpha', 'E_alpha', 'beta', 'e_beta', 'E_beta', 
                 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'P']

    if dir_path != '':
        dir_path += '/'

    file_path = dir_path + 'data/Spectral_parameters_II.txt'

    spectral_data_II = pd.read_csv(file_path, sep=r'\s+', 
                                skiprows=38, names=col_names)
    
    '''
    The dataset has to be rearrange because of the empty spaces in the txt file.
    To fix that, the function takes the 'SMod' column (the spectral model used) that can be 
    'BAND' or 'CPL'.
    
    If SMod is BAND the row of the DataFrame is ok without modifications.
    
    If SMod is CPL the "read_csv" function skipped the empty spaces because we don't have 
    values of beta and correlated errors, so we transpose the values in that row. 
    '''
    
    # Defining the array of f_beta from the dataframe
    arr = spectral_data_II['SMod'].values
    
    # Initializing a loop taking the values in the array and the corresponding indexes
    for index, a in enumerate(arr):
        if a == 'BAND': 
            l1 = []
            l2 = []
            l3 = []
            
        else:
            l1 = ['Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F', 'chi2', 'DoF', 'P']
            l2 = ['beta', 'e_beta', 'E_beta', 'Ep', 'e_Ep', 'E_Ep', 'F', 'e_F', 'E_F']
            l3 = ['beta', 'e_beta', 'E_beta']
            
        # Here I use reversed to not rewrite the values after they changes
        for f1, f2 in zip(list(reversed(l1)), list(reversed(l2))):
            # Transposing the values
            spectral_data_II.at[index, f1] = spectral_data_II.at[index, f2]
        for f3 in l3:
            # Filling the empty spaces with Nan values
            spectral_data_II.at[index, f3] = np.nan
            
        # Rename the values in the Smod column to be consistent with spectral_data_I
        spectral_data_II['SMod'] = spectral_data_II['SMod'].replace('BAND', 'Band')
            
    return spectral_data_II
                
def DataReading(self, dir_path, Energetic=True, Spectral=True):
    '''
    This function uses together the previos functions to read the data from the txt files in 
    the directory path provided by the user.
    
    One can change the input boolean values of 
    "Energetic" and "Spectral" if he want to read only part of the txt file.
    
    After reading the data the function saves it as a pandas DataFrame in the class attributes.
    '''
    if Energetic:
        self.energetic_data_I = ReadEnergeticI(dir_path)
        self.energetic_data_II = ReadEnergeticII(dir_path)
    if Spectral:
        self.spectral_data_I = ReadSpectralI(dir_path)
        self.spectral_data_II = ReadSpectralII(dir_path)

def DatasetExtrapolation(self):
    '''
    This function constructs the dataset with the features (z, DeltaT, alpha_i, beta_i, Ep_i, F_i, 
    alpha_p, beta_p, Ep_p, F_p) and saves it in the class.
    '''
    
    # Using a for because the same procedure applies for I and II data
    dataset_list = []
    for df, BEdata in zip([self.spectral_data_I, self.spectral_data_II], [self.energetic_data_I, self.energetic_data_II]):

        # Splitting the dataset in i and p
        df_i = df[(df['SType'] == 'i')].reset_index(drop=True)
        df_p = df[(df['SType'] == 'p')].reset_index(drop=True)

        # Removing from "i data" the GRBs that are not in "p data" (Because the "i GRBs Dataset" is 
        # bigger than the "p GRBs Dataset").
        # Creating a mask to mark if the i-th element is in the "p dataframe"
        condictions = []
        count = 0
        for i in df_i['ID']:
            condiction = i in df_p['ID'].values
            condictions.append(condiction)

        # Using the mask to drop from "i dataframe" the row corresponding to GRBs that are not in "p dataframe"
        for index, c in enumerate(condictions):
            if not c:
                df_i.drop(index, inplace=True)

        # Selecting Band from the two dataframe ("i" and "p") and CPL (with beta = -2.5) if there isn't Band

        # Filtering the i and p data with Band fit
        df_i_Band = df_i[df_i['SMod'] == 'Band'].reset_index(drop=True)
        df_p_Band = df_p[df_p['SMod'] == 'Band'].reset_index(drop=True)

        # Creating two list to see wich GRBs that are in i or p dataset are not in the filtered dataset with Band fit
        p_list = []
        i_list = []
        for p in df_p['ID']:
            if not p in df_p_Band['ID'].values:
                p_list.append(p)

        for i in df_i['ID']:
            if not i in df_i_Band['ID'].values:
                i_list.append(i)
                
        # Creating two DataFrame with the GRBs of interest in their CPL fit
        df_i_CPL = df_i[df_i['ID'].isin(i_list) & (df_i['SMod'] == 'CPL')]
        df_p_CPL = df_p[df_p['ID'].isin(p_list) & (df_p['SMod'] == 'CPL')]

        # Replace beta with -2.5
        df_i_CPL.loc[:, 'beta'] = -2.5
        df_p_CPL.loc[:, 'beta'] = -2.5


        # Assembling the CPL and Band dataframe sorting by ID for comparing consistency reasons
        df_i_Band_CPL = pd.concat([df_i_Band, df_i_CPL], ignore_index=True)
        df_p_Band_CPL = pd.concat([df_p_Band, df_p_CPL], ignore_index=True)

        # Sorting by ID for comparing consistency reasons
        df_i_Band_CPL_sorted = df_i_Band_CPL.sort_values(by=['ID'])
        df_p_Band_CPL_sorted = df_p_Band_CPL.sort_values(by=['ID'])

        # Resetting the index after the sorting process
        final_df_i = df_i_Band_CPL_sorted.reset_index(drop=True)
        final_df_p = df_p_Band_CPL_sorted.reset_index(drop=True)

        # Rename the columns name of the variables of interest
        final_df_i = final_df_i.rename(columns={'alpha': 'alpha_i', 
                                   'beta': 'beta_i', 
                                   'Ep': 'Ep_i', 
                                   'F': 'F_i'})

        final_df_p = final_df_p.rename(columns={'alpha': 'alpha_p', 
                                   'beta': 'beta_p', 
                                   'Ep': 'Ep_p', 
                                   'F': 'F_p'})

        # Combining the values of interest of the "i data" and the "p data"
        dataset = pd.concat([final_df_i[['ID', 'DeltaT', 'alpha_i', 'beta_i', 'Ep_i', 'F_i']], 
                             final_df_p[['alpha_p', 'beta_p', 'Ep_p', 'F_p']]], axis=1)

        # Extrating the redshift from the Burst Energetics data
        redshift = []
        for name in dataset['ID']:
            temp_df = BEdata[BEdata['ID'] == name]['z'].to_numpy()
            z = temp_df[0]
            redshift.append(z)

        # Creating a dataframe for the redshift
        temp_dict = {'z': redshift}
        z_df = pd.DataFrame(temp_dict)

        # Combining the dataset with the redshift dataframe
        dataset_temp = pd.concat([z_df, dataset[['DeltaT', 'alpha_i', 'beta_i', 'Ep_i', 
                                            'F_i', 'alpha_p', 'beta_p', 'Ep_p', 'F_p']]], 
                            axis=1)
        
        dataset_list.append(dataset_temp)
    
    # Combining the I data with the II data and saving the dataset in the class
    dataset = pd.concat([dataset_list[0], dataset_list[1]], axis=0)
    dataset = dataset.reset_index(drop=True)
    
    self.dataset = dataset
#------------------------------------------------------------------------------------------------#
    
# Just a simple setter
def SetModel(self, model):
    self.model = model
#-------------------------------------------------------------------------------------------------#

# Plotting functions
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

# Model functions
def GetData(self, df):
    '''
    This function takes the data from a pandas dataframe and return the values of X and y 
    to use in the model. This works filtering z for y and then dropping z from the dataframe 
    to have all the features without z for X.
    '''
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

def GridSearch(self, df, grid, scoring = None, refit = True, 
               cv = None, train_size = None, test_size = None):         # work in progress...
    '''
    This function do a Grid Search Cross Validation algorithm to explore the hyperparameters space 
    and find the best hyperparameters combination based on the scoring metrics.
    
    After that the function saves the train and test X and y arrays and the Grid search class 
    from wich one can get the results of the searching procedure.
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
    
    rf_cv = GridSearchCV(pipe, param_grid=grid, scoring=scoring, cv=cv,
                         return_train_score=True, refit=refit, n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    
    # Saving values in the class
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.Grid_search_class = rf_cv