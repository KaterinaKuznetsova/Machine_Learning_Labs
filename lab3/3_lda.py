from math import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split


# функция для вычисления среднего значения
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

# функция для вычисления коэффициента Пирсона
def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff ** 2
        ydiff2 += ydiff ** 2
    if xdiff2 == 0:
        xdiff2 += 0.0000001
    if ydiff2 == 0:
        ydiff2 += 0.0000001

    return diffprod / sqrt(xdiff2 * ydiff2)


def main():
    # загрузка данных
    dataset = pd.read_csv("zoo.data.csv", header=None).values
    animal_attr = dataset[:, 1:-1]  # список атрибутов (признаков) для каждого животного
    animal_class = dataset[:, -1]  # классы животных
    animal_class = animal_class.astype(np.int64, copy=False)
    data_train, data_test, class_train, class_test = train_test_split(animal_attr, animal_class, test_size=0.3,
                                                                      random_state=123)

    print('Количество записей:', animal_attr.shape[0])
    print('Количество признаков:', animal_attr.shape[1])
    print('Количество классов:', len(set(animal_class)))

    # настройки визуализации
    plt.figure(figsize=(10, 8))
    plt.title('Zoo database')
    plt.xlabel('Airborne')
    plt.ylabel('Breathes')

    # визуализация 2х признаков
    for label, marker, color in zip(range(1, 8), ('x', 'o', '^', 'x', 'o', '^', 'x'),
                                    ('blue', 'red', 'green', 'black', 'yellow', 'orange', 'grey')):
        # выбираем признаки airborne и breathes для визуализации
        # вычисляем для каждого класса коэффициент корреляции Пирсона для его признаков
        pearson_coef = pearson_def(animal_attr[:, 5][animal_class == label], animal_attr[:, 10][animal_class == label])
        plt.scatter(x=animal_attr[:, 5][animal_class == label],
                    y=animal_attr[:, 10][animal_class == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {0:}, R={1:.5f}'.format(label, pearson_coef)
                    )
    plt.legend(loc='center')
    plt.show()

    # коэффициент корреляции между признаками airborne и breathes
    print('Коэффициент корреляции между признаками airborne и breathes {} \n'.format(
        pearson_def(animal_attr[:, 5], animal_attr[:, 10])))


    # разбиение классов набора данных с помощью LDA
    sklearn_lda = LDA(n_components=2)
    sklearn_data = sklearn_lda.fit_transform(data_train, class_train)

    # Визуализация разбиения классов после линейного преобразования LDA
    plt.figure(figsize=(10, 8))
    plt.xlabel('vector 1')
    plt.ylabel('vector 2')
    plt.title('Most significant singular vectors after linear transformation via LDA')

    for label, marker, color in zip(range(1, 8), ('x', 'o', '^', 'x', 'o', '^', 'x'),
                                    ('blue', 'red', 'green', 'black', 'yellow', 'orange', 'grey')):
        plt.scatter(x=sklearn_data[:, 0][class_train == label],
                    y=sklearn_data[:, 1][class_train == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label)
                    )

    plt.legend()
    plt.show()

    # классификация с помощью методов LDA и QDA
    for clf in [LDA(), QDA()]:
        clf.fit(data_train, class_train)
        pred_train = clf.predict(data_train)
        pred_test = clf.predict(data_test)
        print(clf, '\n')
        print('Точность классификации на обучающем наборе {:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))
        print('Точность классификации на тестовом наборе {:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))
        print('------------')


main()
