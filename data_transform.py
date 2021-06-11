from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd 

class DFTransform(TransformerMixin, BaseEstimator):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy
        
    def fit(self, *_):
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)
    
def _one_hot_encoding (df, column, classes):
    df[column] = [i if i != None else [] for i in df[column] ] 
    mlb =  MultiLabelBinarizer()
    mlb.fit([[i] for i in classes])
    res = mlb.transform(df[column]) 
    res = pd.DataFrame(mlb.transform(df[column]), columns = [i+'_ohe' for i in classes ] )
    df = df.join(res)
    return df.drop(column, axis=1) 

class RamTransform(TransformerMixin, BaseEstimator): 
    def __init__(self):
        self.column = 'wielkość pamięci ram' 
    
    def fit(self, *_): 
        return self
    
    def transform(self,X):
        return self._ram(X,self.column)  

    def _ram(self, X, column):
        X[column] = [self._to_gb(i) for i in X[column]]
        return X 
    
    def _to_gb(self, x):
        if x != None: 
            int_, unit = x.split(" ")
            if unit == "gb":
                return int(int_)
            if unit == 'mb':
                return int(int_)/1000 

def _matryca(df,column):
    def calculate_matryca (x):
        if x != None:
            return int(x[:2])
        else: 
            return None 
    df[column] = [calculate_matryca(i) for i in df[column]]
    return df 

def _dysk_twardy(df, column):
    df [column] = [i.split(' + ') if i != None else [] for i in df[column]]
    return df 

def _replace(df,column,rep):
    df[column] = [None if i == rep else i for i in df[column]]
    return df 

def _piksele(df, column): 
    def calculate_pixel (x): 
        if (x != None) & (x!='inna'):
            width, height = x.split(" x ")
            if int(width) * int(height) >= 2073600: #=1920 x 1080 full HD 
                return 1
            else: 
                return 0 
        else: 
            return 1 #most_frequent 
    df[column+'_fe'] = [calculate_pixel(i) for i in df[column]]
    return df 

def _seria_procesora (df, column): 
    df[column+'_fe'] = [1 if i in ('intel core i5', 'intel core i7') else 0 for i in df[column]]
    return df 
                  
def _join_column(df, column_list, new_column_name): 
    df[new_column_name+'_fe'] = df[column_list].sum(axis=1)
    return df    
                                  
def _drop(df,column):
    if column in df:
        return df.drop(column, axis=1) 
    else:
        return df