import numpy as np
import matplotlib.pyplot as plt


"""
Описание реализации:

Для каждого класса выбрал одно значение в качестве шаблона.
Далее в методе predict вычислял сумму разности sum(x^2 - t^2) между шаблоном для каждого класса и значением теста.
Из полученных значений выбирал минимальное.
"""


def load_data(path='mnist.npz'): 
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


class MyFirstClassifier(object):
    def __init__(self):
        self.zero_matrix = 0
        self.one_matrix = 0
        self.two_matrix = 0
        self.three_matrix = 0
        self.four_matrix = 0
        self.five_matrix = 0
        self.six_matrix = 0
        self.seven_matrix = 0
        self.eight_matrix = 0
        self.nine_matrix = 0
    def fit(self, x_train, y_train):
        for i in range(0, 300):
            if y_train[i] == 0:
                self.zero_matrix = x_train[i]
            elif y_train[i] == 1:
                self.one_matrix = x_train[i]
            elif y_train[i] == 2:
                self.two_matrix = x_train[i]
            elif y_train[i] == 3:
                self.three_matrix = x_train[i]
            elif y_train[i] == 4:
                self.four_matrix = x_train[i]
            elif y_train[i] == 5:
                self.five_matrix = x_train[i]
            elif y_train[i] == 6:
                self.six_matrix = x_train[i]
            elif y_train[i] == 7:
                self.seven_matrix = x_train[i]
            elif y_train[i] == 8:
                self.eight_matrix = x_train[i]
            elif y_train[i] == 9:
                self.nine_matrix = x_train[i]
    def predict(self, x_test):
        result = []
        for i in range(len(x_test)):
            temp = []
            temp.append(np.sum(self.zero_matrix ** 2 - x_test[i] ** 2)) # 0
            temp.append(np.sum(self.one_matrix ** 2 - x_test[i] ** 2)) # 1
            temp.append(np.sum(self.two_matrix ** 2 - x_test[i] ** 2)) # 2
            temp.append(np.sum(self.three_matrix ** 2 - x_test[i] ** 2)) # 3
            temp.append(np.sum(self.four_matrix ** 2 - x_test[i] ** 2)) # 4
            temp.append(np.sum(self.five_matrix ** 2 - x_test[i] ** 2)) # 5
            temp.append(np.sum(self.six_matrix ** 2 - x_test[i] ** 2)) # 6
            temp.append(np.sum(self.seven_matrix ** 2 - x_test[i] ** 2)) # 7
            temp.append(np.sum(self.eight_matrix ** 2 - x_test[i] ** 2)) # 8
            temp.append(np.sum(self.nine_matrix ** 2 - x_test[i] ** 2)) # 9
            result.append(np.argmin(temp))
        return result
    
def accuracy_score(pred, gt):
    return np.mean(pred==gt)


(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
cls = MyFirstClassifier()
cls.fit(x_train, y_train)
pred = cls.predict(x_test)

print('accuracy is %.4f' % accuracy_score(pred, y_test))
