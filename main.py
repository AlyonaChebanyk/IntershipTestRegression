import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# отримання тестових даних
def get_test_data():
    return pd.read_csv('internship_hidden_test.csv')


# отримання даних для тренування
def get_train_data():
    df = pd.read_csv('internship_train.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


# отримання моделі
def get_random_forest_regressor():
    return RandomForestRegressor(n_estimators=10)


if __name__ == '__main__':
    X, y = get_train_data()
    model = get_random_forest_regressor()

    # тренування моделі
    model.fit(X, y)

    # отримання передбачення
    y_pred = model.predict(get_test_data())

    # збереження результатів у файлі result.csv
    pd.DataFrame(y_pred).to_csv('result.csv', header=['Prediction'], index=False)
