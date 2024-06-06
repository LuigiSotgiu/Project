from types import MethodType
from functions import *

class ML_GRB:
    def __init__(self):
        self.energetic_data = None
        self.spectral_data = None

        self.model = None

        self.SetModel = MethodType(SetModel, self)
        self.DataReading = MethodType(DataReading, self)
        self.CorrMatrix = MethodType(CorrMatrix, self)

class RND_FOREST:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.forest = None
        
        self.GetData = MethodType(GetData, self)
        self.Run = MethodType(Run, self)