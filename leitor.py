"""
Caleta de dados para aprendizagem
"""
import sys
import logging
import pandas as pd
from pandas import Series, DataFrame

from sklearn import tree
# import numpy as np
# x=pd.DataFrame(np.array([['a','a'], [1,2]]))

A1 = 0
B2 = 1


def read_data():
    """ - """
    try:

        features = []
        jdata = pd.read_json("./data.json")

        for item in jdata.datarobotic:
            if item['stratification']['class'] == "A1":

                features.append([A1, item['stratification']['points']])
            elif item['stratification']['class'] == "B2":

                features.append([B2, item['stratification']['points']])

        print('features:', features)

        idade = [21, 22, 20, 23]

        # Classificação árvore de decisão
        clf = tree.DecisionTreeClassifier()

        # Identifica padrões
        clf = clf.fit(features, idade)

        # Executa aprendizagem conforme parâmetros de entrada
        resultado = clf.predict([[1, 25]])

        print('resultado:', resultado)

    except:
        logging.exception(sys.exc_info()[0])  # LANÇA EXCEPTION


if __name__ == '__main__':

    logging.info('Chegamos aqui')
    # DATA as pandas.core.frame.DataFrame
    read_data()
