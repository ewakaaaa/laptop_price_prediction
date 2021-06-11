import numpy as np 
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.cluster import KMeans

def check_missing(df):
    for column in df.columns:
        missing = column, df[column].isnull().sum() *100/ df.shape[0]
        if missing[1] == 0: continue
        print(missing)

def get_one_hot_encoding_features(df,list_):
    one_hot_feats = [i+'_ohe' for i in list_ ]
    feats = df.select_dtypes(include=[np.int64, np.float64]).columns  
    feats = [feat for feat in feats if feat in one_hot_feats]
    return feats

def get_numeric_features(df,black_list):
    black_list = black_list + ['buynow_price']
    feats = df.select_dtypes(include=[np.int64, np.float64]).columns 
    feats = [feat for feat in feats if feat not in black_list]
    return feats 

def train_and_predict(model, X, y):
    model.fit(X, y)
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred))

def get_feature_names(steps):
    feature_names = [] 
    for i in range(0,len(steps)):
        feature_names = feature_names + list(steps[i][2])
    return feature_names 

def run_cv(model, X, y, folds=4, target_log=False,cv_type=KFold):
    cv = cv_type(n_splits=folds)
    
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if target_log:
            y_train = np.log(y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if target_log:
            y_pred = np.exp(y_pred)
            y_pred[y_pred < 0] = 0 #czasem może być wartość ujemna

        score = np.sqrt(mean_squared_error(y_test, y_pred)) 
        scores.append( score )
        
    return np.mean(scores), np.std(scores)

def plot_learning_curve(model, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), target_log=False):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    if target_log:
        y = np.log(y)
    
    def my_scorer(model, X, y):
        y_pred = model.predict(X)
        
        if target_log:
            y = np.exp(y)
            y_pred = np.exp(y_pred)
            y_pred[ y_pred<0 ] = 0
        
        return np.sqrt(mean_squared_error(y, y_pred)) 

        
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=my_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_feature_importances(model, feats, limit = 20):
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1][:limit]

    plt.figure(figsize=(20, 8))
    plt.title("Feature importances")
    plt.bar(range(limit), importances[indices])
    plt.xticks(range(limit), [feats[i] for i in indices], rotation='vertical')
    plt.show()
  
    
def get_models():
    return [
        ('LinearRegression', LinearRegression()),
        ('DecisionTree', DecisionTreeRegressor(max_depth=10)),
        ('RandomForest' ,RandomForestRegressor(max_depth=10, n_estimators=20)),
        ('XGB',xgb.XGBRegressor(objective='reg:squarederror'))
    ]

def run(X,y, plot_lc=False, folds=3, ylim=(0, 2), target_log=False):
    for model_name, model in get_models():
        score_mean, score_std = run_cv(model, X, y, folds=folds, target_log=target_log)
        print("[{0}]: {1} +/-{2}".format(model_name, score_mean, score_std))
        sys.stdout.flush() #wypisujemy wynik natychmiast, bez buforowania

        if False == plot_lc: continue
        plt = plot_learning_curve(model, model_name, X, y)
        plt.show()


def calculate_WSS(X, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k).fit(X)
        centroids = kmeans.cluster_centers_
        y_pred = kmeans.predict(X)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(X)):
            curr_center = centroids[y_pred[i]]
            curr_sse += (X[i, 0] - curr_center[0]) ** 2 + (X[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse