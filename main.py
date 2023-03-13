import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


def get_test_data():
    return pd.read_csv('internship_hidden_test.csv')


def get_train_data():
    df = pd.read_csv('internship_train.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def get_linear_regression_model():
    return LinearRegression()


def get_random_forest_regressor():
    return RandomForestRegressor(n_estimators=5)


def get_gradient_boosting_regressor():
    return GradientBoostingRegressor()


def get_ridge():
    return Ridge()


def get_svr():
    return SVR()


def get_scores(X, y, model):
    return cross_val_score(model, X, y, scoring='neg_root_mean_squared_error')


if __name__ == '__main__':
    X, y = get_train_data()
    model = get_random_forest_regressor()
    model.fit(X, y)
    y_pred = model.predict(get_test_data())
    pd.DataFrame(y_pred).to_csv('result')
    # scores = get_scores(X, y, model)
    # print('Linear Regression model')
    # print('RMSE:', -scores)
    # print('Середнє RMSE:', -scores.mean())
