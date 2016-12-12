import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data(filename):
    categ = ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
             'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
    train_df = pd.read_csv(filename, header=None, names=categ)
    train_data = train_df.values
    animal_attr = train_data[:, 1:-1]
    animal_class = train_data[:, -1]
    animal_class = animal_class.astype(np.int64, copy=False)
    return animal_attr, animal_class


def classificate(train_sizes):
    result = []
    class_list = [RandomForestClassifier(), DecisionTreeClassifier()]
    animal_attr, animal_class = load_data('zoo.data.csv')

    for train_size in train_sizes:
        result.append(str(int(train_size * 100)) + "%")
        data_train, data_test, class_train, class_test = train_test_split(animal_attr, animal_class,
                                                                          test_size=1 - train_size)
        for clf in class_list:
            clf.fit(data_train, class_train)
            result.append(clf.score(data_test, class_test))
    return result


def print_result():
    for k in classificate([0.6, 0.7, 0.8, 0.9]):
        print(k)


print_result()
