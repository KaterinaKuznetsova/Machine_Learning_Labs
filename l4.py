import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Чтение датасета
train_data = pd.read_csv('zoo.data.csv', header=None).values
animal_attr = train_data[:, 1:-1]
animal_class = train_data[:, -1]
animal_class = animal_class.astype(np.int64, copy=False)
data_train, data_test, class_train, class_test = train_test_split(animal_attr, animal_class, test_size=0.4)


# Классификация и вычисление точности
def clf_acc(clf):
    clf.fit(data_train, class_train)
    accuracy = clf.score(data_test, class_test)
    return accuracy


clf_list = ['linear', 'rbf', 'poly', 'sigmoid']

# Тестируем различные параметры:

# 1. Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
# {kernel : string, optional (default=’rbf’)}

for kern in clf_list:
    print('Тип ядра: {}, Точность: {:.9f}'.format(kern, clf_acc(SVC(kernel=kern))))
print('-------------********-------------')

# Для датасета zoo.data наиболее точной оказалась линейная функция ядра.
# 2-е место: Полиномиальная функция
# 3-е место: Радиальная базисная функция
# 4-е место: Сигмоидная функция

# 2. Penalty parameter C of the error term. {C : float, optional (default=1.0)}
penalties = np.arange(0.1, 10, 0.5)
for value in penalties:
    print('Штраф за ошибку: {}, Точность: {:.9f}'.format(value, clf_acc(SVC(C=value))))
print('-------end penalties-------')

# В ходе испытаний было замечено, что чем больше параметр penalty, тем выше точность.

# 3. Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. {degree : int, optional (default=3)}

degrees = np.arange(1, 10, 1)
for value in degrees:
    print('Степень полинома: {}, Точность: {:.9f}'.format(value, clf_acc(SVC(kernel='poly', degree=value))))
print('-------end degrees-------')
# Самая высокая точность классификации получается при степени от 2 до 4

# 4. Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. {gamma : float, optional (default=’auto’)}

gammas = np.arange(0.01, 1, 0.05)
for kern in clf_list[1:]:
    print('Тип ядра: {}'.format(kern))
    for value in gammas:
        print('Коэффициент ядра: {:.3f}, Точность: {:.9f}'.format(value, clf_acc(SVC(kernel=kern, gamma=value))))
print('-------end gammas-------')

# При увеличении параметра gamma,
# для полиномиального ядра точность возрастает до показателей линейной функции (87,8048780%)
# rbf точность возрастает, а потом снова падает
# sigmoid точность уменьшается

# 5. Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. {coef0 : float, optional (default=0.0)}

coef0 = np.arange(0.1, 1, 0.1)
for kern in clf_list[2:]:
    print('Тип ядра: {}'.format(kern))
    for value in coef0:
        print('Независимый терм: {:.1f}, Точность: {:.9f}'.format(value, clf_acc(SVC(kernel=kern, coef0=value))))
    print('-------end coef0-------')
# Самым эффективным из протестированных является параметр 0.9 для poly и 0.1 для sigmoid


# 6. Whether to enable probability estimates. { probability : boolean, optional (default=False) }
probabilities = [False, True]
for value in probabilities:
    print('Оценка вероятности: {}, Точность: {:.9f}'.format(value, clf_acc(SVC(probability=value))))
print('-------end probability-------')

# 7. Whether to use the shrinking heuristic. { shrinking : boolean, optional (default=True) }
shrinkings = [False, True]
for value in shrinkings:
    print('Shrinking heuristic: {}, Точность: {:.9f}'.format(value, clf_acc(SVC(probability=value))))
print('-------end shrinking-------')

# 8. Tolerance for stopping criterion. { tol : float, optional (default=1e-3) }

tols = [1e-1, 1e-2, 1e-3, 1e-4]
for value in tols:
    print('tol: {}, Точность: {:.9f}'.format(value, clf_acc(SVC(tol=value))))
print('-------end tol-------')
# Параметры probability, shrinkings и tol на точность никак не повлияли


# Применим оптимальные параметры для SVM
print('Тип ядра: linear, Точность: {:.9f}'.format(clf_acc(SVC(kernel='linear'))))
clf = SVC(C=8, kernel='poly', degree=4, gamma=1, coef0=0.9)
print('Accuracy: {:.9f}'.format(clf_acc(clf)))

clf = SVC(C=7, kernel='rbf', degree=3, gamma=0.5)
# print('{}'.format(clf))
print('Accuracy: {:.9f}'.format(clf_acc(clf)))
