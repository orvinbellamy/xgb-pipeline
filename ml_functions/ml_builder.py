### Machine Learning Builder ###
import numpy as np
import random
import seaborn as sns
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score
import xgboost as xgb

## Classes
class TrainTestSplit:
    def __init__(self, X_train, y_train, X_test, y_test, class_ratio):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_ratio = class_ratio
        
class XGBParam:
    def __init__(self, learning_rate, n_estimators, subsample, colsample_bytree, gamma, min_child_weight):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        
        
## Functions
def eda_plot(df:pd.DataFrame, x_axis, y_axis, log_x=None, log_y=None):
    df_plot = df[[x_axis, y_axis]]
    
    if log_x is not None:
        df_plot[x_axis] = np.log(df_plot[x_axis] + log_x)
        
    if log_y is not None:
        df_plot[y_axis] = np.log(df_plot[y_axis] + log_y)
        
    # using ci argument makes the plot takes way longer
    sns.regplot(data=df_plot, x=x_axis, y=y_axis, scatter=True, logistic=True, ci=None, truncate=True, line_kws={'color': 'red'})
    
def sqrt_neg(df_plc:pd.DataFrame, col):
    
    col_name = col + "_sqrt"
    df_plc[col_name] = np.sqrt(df_plc[col].abs())
    
    # Add negative
    df_plc.loc[df_plc[col]<0, col_name] = df_plc.loc[df_plc[col]<0, col_name] * (-1)

def create_default_transformer(X_train):
    ## Column Transformer
    
    # creating the column tansformer
    # note be careful this is filtering categorical column using 'object' dtypes
    # in the case where this might get mixed up with non-categorical, edit this next line
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_features = X_train.select_dtypes(include=['float', 'int']).columns

    # create categorical transformer
    # add imputation if needed

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(sparse=False, handle_unknown='ignore'))]
    )

    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler())
            ]
    )

    # create column transformer
    col_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numerical_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return col_transformer

def create_xgb_model(df: pd.DataFrame, col_features: list, col_label: str, 
	test_size: float = 0.2, random_seed=123, model_param: dict = {}):
    
    ## Configurations
    
    # Create col_train
    col_train = col_features + [col_label]
    
    # Set default parameters
    default_param = {'learning_rate': 0.05, 'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'min_child_weight': 1, 'alpha': 0.01}
    
    for key, value in default_param.items():
        model_param.setdefault(key, value)
    
    ## Train-test split
    df_train, df_test = train_test_split(df[col_train], test_size=test_size, random_state=random_seed)

    X_train = df_train.drop(columns=col_label)
    y_train = df_train[col_label]
    X_test = df_test.drop(columns=col_label)
    y_test = df_test[col_label]

    class_ratio = sum(y_train == 0) / sum(y_train == 1)
    
    col_transformer = create_default_transformer(X_train=X_train)
    
    xgb_model = Pipeline(
        steps=[('columntransformer', col_transformer),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=class_ratio, learning_rate=model_param['learning_rate'], n_estimators=model_param['n_estimators'], max_depth=10, subsample=model_param['subsample'], colsample_bytree=model_param['colsample_bytree'], gamma=model_param['gamma'], alpha=model_param['alpha'], min_child_weight=model_param['min_child_weight'], random_state=random_seed))]
    )

    # # Train and test the model
    xgb_model.fit(X_train, y_train)
    xgb_testscore = xgb_model.score(X_test, y_test)
    print('rf test score: ', xgb_testscore)
    
    return xgb_model, df_test

def xgb_random_optimize(df, col_features, col_label, param, test_size=0.2, random_seed=123, cv=3, n_iter=100):

    ## XGB parameter hyperoptimization
    
    ## Train-test split
    
    col_train = col_features + [col_label]
    
    df_train, df_test = train_test_split(df[col_train], test_size=test_size, random_state=random_seed)

    X_train = df_train.drop(columns=col_label)
    y_train = df_train[col_label]
    X_test = df_test.drop(columns=col_label)
    y_test = df_test[col_label]

    class_ratio = sum(y_train == 0) / sum(y_train == 1)
    
    col_transformer = create_default_transformer(X_train=X_train)
    
    # Create dummy pipeline
    pipe_xgb = Pipeline(
        steps=[('columntransformer', col_transformer),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=class_ratio, max_depth=10, random_state=123))]
    )

    # Set random_search
    xgb_grid = RandomizedSearchCV(pipe_xgb, random_state=random_seed, param_distributions = param, return_train_score=True, cv=cv, n_iter=n_iter, scoring='precision')
    xgb_grid.fit(X_train, y_train)

    xgb_grid_result = pd.DataFrame(xgb_grid.cv_results_)[["params", "mean_test_score", "mean_train_score", "rank_test_score"]].sort_values(by='rank_test_score')

    # get best parameter
    best_param = {} 
    best_param['learning_rate'] = xgb_grid.best_params_['xgb__learning_rate']
    best_param['n_estimators'] = xgb_grid.best_params_['xgb__n_estimators']
    best_param['subsample'] = xgb_grid.best_params_['xgb__subsample']
    best_param['colsample_bytree'] = xgb_grid.best_params_['xgb__colsample_bytree']
    best_param['gamma'] = xgb_grid.best_params_['xgb__gamma']
    best_param['min_child_weight'] = xgb_grid.best_params_['xgb__min_child_weight']

    # Create new pipeline with optimized hyperparameter
    xgb_model = Pipeline(
        steps=[('columntransformer', col_transformer),
            ('xgb', xgb.XGBClassifier(
                objective='binary:logistic', 
                scale_pos_weight=class_ratio, 
                learning_rate=best_param['learning_rate'], 
                n_estimators=best_param['n_estimators'], 
                subsample=best_param['subsample'], 
                colsample_bytree=best_param['colsample_bytree'],
                gamma=best_param['gamma'], 
                min_child_weight=best_param['min_child_weight'],
                max_depth=15, 
                random_state=random_seed))]
    )

    # # Train and test the model
    xgb_model.fit(X_train, y_train)
    xgb_testscore = xgb_model.score(X_test, y_test)
    print('rf test score: ', xgb_testscore)
    
    return xgb_model, xgb_grid_result, best_param, df_test

def xgb_random_feature_selection(df: pd.DataFrame, fixed_features: list, trial_features: list, num_test_features: int, label: str, random_seed: int, model_param: dict = {}, n_iter: int = 100, test_size: float = 0.2):

    # Set default parameters
    default_param = {'learning_rate': 0.05, 'n_estimators': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'min_child_weight': 1, 'alpha': 0.01}
    
    for key, value in default_param.items():
        model_param.setdefault(key, value)
    
    dic_feature_loop = {'iter': [], 'numeric_features': [], 'test_score': [], 
                        'precision_09': [], 'precision_08': []}

    # Loop to iterate X number of times
    for i in range(0, n_iter):
        
        # Select N random items from the original list
        random_features = random.sample(trial_features, num_test_features)
        
        col_train_loop = fixed_features + random_features + [label]
        
        df_train, df_test = train_test_split(df[col_train_loop], test_size=test_size, random_state=random_seed)

        X_train = df_train.drop(columns=label)
        y_train = df_train[label]
        X_test = df_test.drop(columns=label)
        y_test = df_test[label]
        
        class_ratio = sum(y_train == 0) / sum(y_train == 1)
        
        col_transformer = create_default_transformer(X_train=X_train)
        
        xgb_model = Pipeline(
        steps=[('columntransformer', col_transformer),
            ('xgb', xgb.XGBClassifier(
                objective='binary:logistic', 
                scale_pos_weight=class_ratio,
                learning_rate=model_param['learning_rate'], 
                n_estimators=model_param['n_estimators'],
                max_depth=15, 
                subsample=model_param['subsample'],
                colsample_bytree=model_param['colsample_bytree'],
                gamma=model_param['gamma'],
                alpha=0.01,
                min_child_weight=model_param['min_child_weight'],
                random_state=random_seed
                )
            )]
        )
        
        xgb_model.fit(X_train, y_train)
        xgb_testscore = xgb_model.score(X_test, y_test)
        
        df_test['prediction'] = xgb_model.predict_proba(df_test)[:, 1]
        
        df_test['prediction_factor'] = 0
        df_test.loc[(df_test['prediction'] >= 0.90), 'prediction_factor'] = 1
        
        precision_09 = precision_score(df_test['prediction_factor'], df_test[label])
        
        df_test.loc[(df_test['prediction'] >= 0.80), 'prediction_factor'] = 1
        
        precision_08 = precision_score(df_test['prediction_factor'], df_test[label])
        
        dic_feature_loop['iter'].append(i)
        dic_feature_loop['numeric_features'].append(random_features)
        dic_feature_loop['test_score'].append(xgb_testscore)
        dic_feature_loop['precision_09'].append(precision_09)
        dic_feature_loop['precision_08'].append(precision_08)
        
    df_feature_loop = pd.DataFrame.from_dict(dic_feature_loop)
    
    return df_feature_loop