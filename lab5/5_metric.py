import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Чтение датасета
train_data = pd.read_csv('zoo.data.csv', header=None).values
animal_attr = train_data[:, 1:-1]
animal_class = train_data[:, -1]
animal_class = animal_class.astype(np.int64, copy=False)
X_train, X_test, Y_train, Y_test = train_test_split(animal_attr, animal_class, test_size=0.35)


def calc_metric(estimator):
    # для каждого метода классификации
    print('Классификатор: {}'.format(estimator))
    # вычисляем точность классификации и логарифм функции правдоподобия
    for scoring in ['accuracy', 'neg_log_loss']:
        result = cross_val_score(estimator, animal_attr, animal_class, cv=KFold(n_splits=2, shuffle=True),
                                 scoring=scoring)
        print(' Метрика: {}. Точность: {:.3f} ({:.3f})'.format(scoring, result.mean(), result.std()))

    # обучаем классификатор
    estimator.fit(X_train, Y_train)
    # классифицируем тестовую выборку
    predicted = estimator.predict(X_test)

    print(' Метрика: Confusion Matrix\n {}'.format(confusion_matrix(Y_test, predicted)))
    print(' Метрика: Classification Report\n {}'.format(classification_report(Y_test, predicted)))
    print('------------------\n')


calc_metric(GaussianNB())
calc_metric(KNeighborsClassifier(n_neighbors=25))
