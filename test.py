import os
import pandas as pd
from ML_GRB import ML_GRB, RND_FOREST

dir_path = os.path.dirname(os.path.abspath(__file__))   # directory path

obj = ML_GRB()      # initializing the main class ---> see ML_GRB.py
obj.DataReading(dir_path)       # reading the data ---> see functions.py

# Taking the dataframes from the class
spec_par = obj.spectral_data
energ_par = obj.energetic_data

# Defining the features to use in the model from the two dataframes
filter_ener = ['z', 'S', 'Tp64', 'Fp64', 'Eiso','Liso']
filter_spec = ['tstart', 'DeltaT', 'alpha', 'Ep', 'F']

spec_par_reset = spec_par.reset_index(drop=True)  # Reset index and drop the old index column

# Concatenating the two dataframe to form one and filtering
df1 = energ_par[energ_par['ID'] != 'GRB080413B'][filter_ener]   # Exluding GRB080413B becuase 'SMOD' = 'PL'
df2 = spec_par[(spec_par['SMod'] == 'CPL') & (spec_par['SType'] == 'i')][filter_spec]    # Using the CPL SMod
df_temp = pd.concat([df1, df2], axis=0)     # Concatenating
df = df_temp.apply(lambda x: pd.Series(x.dropna().values))      # Dropping the Nan values

# Selecting the features to use (one can change directly from here if he wants)
features = ['z', 'S', 'Tp64', 'Fp64', 'Eiso','Liso', 
          'tstart', 'DeltaT', 'alpha', 'Ep', 'F']
dataset = df[features]

# Inizialising the model class ---> see ML_GRB.py
model = RND_FOREST()
obj.SetModel(model)     # Setting the model in the main class

# Fitting the model ---> see functions.py
obj.model.Run(dataset, train_size = 0.66, random_state = 42, n_estimators = 1000)

# Printing the results
test_score = obj.model.forest.score(obj.model.X_test, obj.model.y_test)
train_score = obj.model.forest.score(obj.model.X_train, obj.model.y_train)

for a, b in zip(obj.model.y_test, obj.model.y_pred):
    print('z test: {:1.3f}\t\tz predicted: {:1.3f}'.format(a, b))
    
print('The test score is : {:.3f} while the train score is: {:.3f}'.format(test_score, train_score))