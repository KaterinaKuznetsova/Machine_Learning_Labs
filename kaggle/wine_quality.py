import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier


def load_train_data():
    train_df = pd.read_csv('training_classification_regression_2015.csv', header=0)
    # print('Количество записей в тренировочном датасете: %d, количество атрибутов: %d' % train_df.shape)

    # разбиение обучающего датасета на значения атрибутов и массив классов
    train_wine_attr = train_df.drop(['quality', 'type'], axis=1).values
    train_wine_quality = train_df['quality'].astype(int).values

    # нормализация значений атрибутов
    scaler = preprocessing.StandardScaler().fit(train_wine_attr)
    train_wine_attr = scaler.transform(train_wine_attr)

    return train_wine_attr, train_wine_quality, scaler


def load_test_data(scaler):
    test_df = pd.read_csv('challenge_public_test_classification_regression_2015.csv', header=0)
    # print('Количество записей в тестовом датасете: %d, количество атрибутов: %d' % test_df.shape)

    # удаление ненужных атрибутов тестового массива
    test_wine_attr = test_df.drop(['id', 'quality', 'type'], axis=1).values

    # нормализация значений атрибутов
    test_wine_attr = scaler.transform(test_wine_attr)

    return test_wine_attr, test_df['id']


def test_class_models(train_wine_attr, train_wine_quality):
    class_models = [DummyClassifier(), AdaBoostClassifier(), BaggingClassifier(), ExtraTreesClassifier(),
                    GradientBoostingClassifier(), RandomForestClassifier(), LogisticRegression(),
                    PassiveAggressiveClassifier(), RidgeClassifier(), SGDClassifier(), GaussianNB(),
                    KNeighborsClassifier(), NearestCentroid(), MLPClassifier(),
                    LabelPropagation(), SVC(), LinearSVC(), DecisionTreeClassifier(),
                    ExtraTreeClassifier()]

    for clf in class_models:
        result = cross_val_score(clf, train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
        print('Метод классификации: {}.\n  Точность: {:.5f} ({:.5f})'.format(clf, result.mean(), result.std()))


def test_LabelPropagation(train_wine_attr, train_wine_quality):
    print('Тестирование LabelPropagation')

    # Без параметров
    result = cross_val_score(LabelPropagation(), train_wine_attr, train_wine_quality, cv=KFold(n_splits=20))
    print('Точность без параметров: {:.5f} ({:.5f})'.format(result.mean(), result.std()))

    # Функция ядра
    # knn
    for i in [10, 20, 30, 50, 60]:
        res = cross_val_score(LabelPropagation(kernel='knn', n_neighbors=i), train_wine_attr, train_wine_quality,
                              cv=KFold(n_splits=20))
        print('kernel = knn, n_neighbors =', i, res.mean(), res.std())
    # rbf
    res = cross_val_score(LabelPropagation(kernel='rbf'), train_wine_attr, train_wine_quality,
                          cv=KFold(n_splits=20))
    print('kernel = rbf', res.mean(), res.std())
    print('--------------')

    # Параметр для rbf ядра
    gammas = [0.1, 0.5, 1, 5, 10, 20, 30, 40]
    for gamma in gammas:
        res = cross_val_score(LabelPropagation(kernel='rbf', gamma=gamma), train_wine_attr, train_wine_quality,
                              cv=KFold(n_splits=20))
        print('gamma =', gamma, res.mean(), res.std())
    print('--------------')

    # Параметр для rbf ядра
    alphas = [0.01, 0.1, 0.5, 1, 2, 3]
    for alpha in alphas:
        res = cross_val_score(LabelPropagation(alpha=alpha), train_wine_attr, train_wine_quality,
                              cv=KFold(n_splits=20))
        print('alpha =', alpha, res.mean(), res.std())
    print('--------------')


def test_ExtraTreesClassifier(train_wine_attr, train_wine_quality):
    print('Тестирование ExtraTreesClassifier')

    # Без параметров
    result = cross_val_score(ExtraTreesClassifier(), train_wine_attr, train_wine_quality, cv=KFold(n_splits=20))
    print('Точность без параметров: {:.5f} ({:.5f})'.format(result.mean(), result.std()))

    # Количество деревьев (default=10)
    n_estimators = np.arange(10, 100, 10)
    for n_estimator in n_estimators:
        result = cross_val_score(ExtraTreesClassifier(n_estimators=n_estimator), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('n_estimators =', n_estimator, result.mean(), result.std())
    print('--------------')

    # Функция оценки качества разбиения (default=”gini”)
    criterion = ['gini', 'entropy']
    for cr in criterion:
        result = cross_val_score(ExtraTreesClassifier(criterion=cr), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('criterion =', cr, result.mean(), result.std())
    print('--------------')

    # Количество фич при поиске наилучшего разбиения (default=”auto”)
    max_features = ['auto', 'log2', 1, 3, 5, 7, 9, 11]
    for features in max_features:
        result = cross_val_score(ExtraTreesClassifier(max_features=features), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('max_features =', features, result.mean(), result.std())
    print('--------------')

    # Использовать ли предыдушие решения? (default=False)
    warm_start = [True, False]
    for war_start in warm_start:
        result = cross_val_score(ExtraTreesClassifier(warm_start=war_start), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('warm_start =', war_start, result.mean(), result.std())
    print('--------------')


def test_RandomForestClassifier(train_wine_attr, train_wine_quality):
    print('Тестирование RandomForestClassifier')

    # Без параметров
    result = cross_val_score(RandomForestClassifier(), train_wine_attr, train_wine_quality, cv=KFold(n_splits=20))
    print('Точность без параметров: {:.5f} ({:.5f})'.format(result.mean(), result.std()))

    # Количество деревьев (default=10)
    n_estimators = np.arange(60, 200, 20)
    for n_estimator in n_estimators:
        result = cross_val_score(RandomForestClassifier(n_estimators=n_estimator), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('n_estimators =', n_estimator, result.mean(), result.std())
    print('--------------')

    # Функция оценки качества разбиения (default=”gini”)
    criterions = ['gini', 'entropy']
    for criterion in criterions:
        result = cross_val_score(RandomForestClassifier(criterion=criterion), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('criterion =', criterion, result.mean(), result.std())
    print('--------------')

    # Количество фич при поиске наилучшего разбиения (default=”auto”)
    max_features = ['auto', 'log2', 1, 3, 5, 7, 9, 11]
    for features in max_features:
        result = cross_val_score(RandomForestClassifier(max_features=features), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('max_features =', features, result.mean(), result.std())
    print('--------------')

    # Использовать ли предыдушие решения? (default=False)
    warm_start = [True, False]
    for war_start in warm_start:
        result = cross_val_score(RandomForestClassifier(warm_start=war_start), train_wine_attr, train_wine_quality,
                                 cv=KFold(n_splits=20))
        print('warm_start =', war_start, result.mean(), result.std())
    print('--------------')


def inspect_3(train_wine_attr, train_wine_quality):
    test_LabelPropagation(train_wine_attr, train_wine_quality)
    print('///////////////////////////')

    test_ExtraTreesClassifier(train_wine_attr, train_wine_quality)
    print('///////////////////////////')

    test_RandomForestClassifier(train_wine_attr, train_wine_quality)
    print('///////////////////////////')



def test_3(train_wine_attr, train_wine_quality):
    print('Подбор оптимальных параметров LabelPropagation')
    result = cross_val_score(LabelPropagation(kernel='rbf', gamma=20, alpha=1), train_wine_attr, train_wine_quality,
                             cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))


    print('\nПодбор оптимальных параметров ExtraTreesClassifier')
    result = cross_val_score(ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_features=3),
                             train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))
    result = cross_val_score(
        ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_features='log2', warm_start=True),
        train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))
    result = cross_val_score(
        ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_features='log2', warm_start=False),
        train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))


    print('\nПодбор оптимальных параметров RandomForestClassifier')
    result = cross_val_score(
        RandomForestClassifier(n_estimators=180, max_features=1, criterion='entropy', warm_start=True),
        train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))
    result = cross_val_score(RandomForestClassifier(n_estimators=180, criterion='entropy', warm_start=False),
                             train_wine_attr, train_wine_quality, cv=KFold(n_splits=50))
    print('Точность: {:.5f} ({:.5f})'.format(result.mean(), result.std()))


def train():
    train_wine_attr, train_wine_quality, s = load_train_data()
    test_wine_attr, id = load_test_data(s)

    clf = ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_features='log2', warm_start=True)
    clf.fit(train_wine_attr, train_wine_quality)
    result = clf.predict(test_wine_attr)

    # Создание выходного файла
    output = pd.DataFrame(id)
    output['quality'] = result
    output.to_csv('submission.csv', index=False)


def main():
    train_wine_attr, train_wine_quality, s = load_train_data()

    # inspect_3(train_wine_attr, train_wine_quality)
    test_3(train_wine_attr, train_wine_quality)


main()
