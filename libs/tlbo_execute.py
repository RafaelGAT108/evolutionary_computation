from tlbo import TLBO
from dataclasses import dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, make_scorer, log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.optimizers import RMSprop
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


def run_model(model, X_train, y_train, X_test, y_test, metric):

    if metric == 'train':
        loss_scorer = make_scorer(log_loss, response_method="predict_proba")
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring=loss_scorer)
    
    elif metric == 'validation':
        scores = cross_val_score(model, X_train, y_train, cv=3)

    mean_loss = scores.mean()
    model.fit(X_train, y_train)
    mean_loss = round(mean_loss, 4)

    y_pred = model.predict(X_test)
    y_test = to_categorical(y_test, num_classes=6)

    test_accuracy = accuracy_score(y_test, y_pred)

    return scores, mean_loss, test_accuracy


def evaluate_function(chromossome, metric = 'train'):
    data = pd.read_csv('/home/rafael/Mestrado/Artigo TLBO/database/mulher1.csv', delimiter=';')
    label_counts = data['Output'].value_counts().sort_index()
    data['Output'] = data['Output'] - 1
    data = data.dropna(axis=1)
    colunas_com_infinitos = data.columns[data.isin([np.inf, -np.inf]).any()]
    data = data.drop(columns=colunas_com_infinitos)

    y = data['Output']
    X = data.drop(columns=['Output'], inplace=False)

    X = X.loc[:, np.array(chromossome, dtype=bool)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    def create_model():
        model = Sequential([
            Dense(10, activation='tanh', input_shape=(X_train.shape[1],)),
            Dense(len(np.unique(y)), activation='softmax')
        ])
        optimizer = RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model_mlp = KerasClassifier(build_fn=create_model, epochs=20, verbose=0)

    y_train = to_categorical(y_train, num_classes=6)
    (scores_knn, mean_loss_knn, test_accuracy) = run_model(model=model_mlp, 
                                                           X_train=X_train, 
                                                           y_train=y_train, 
                                                           X_test=X_test, 
                                                           y_test=y_test,
                                                           metric=metric)
    
    features = list(chromossome).count(1)
    if metric == 'train':
        return mean_loss_knn + 0.001*features
        
    elif metric == 'validation':
        print(f'Accuracy in Training: {mean_loss_knn}')
        print(f'Accuracy in Validation: {round(test_accuracy, 4)}')
        return mean_loss_knn


if __name__ == '__main__':
    tlbo = TLBO(cost_function=evaluate_function,
            population_size=20,
            dim=634,
            iterations_size=20)
    
    best_student = tlbo.execute()
    evaluate_function(best_student.position, metric = 'validation')
    print(best_student.position)