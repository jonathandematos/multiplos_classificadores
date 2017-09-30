#!/usr/bin/python
#
from __future__ import print_function
import arff
import sys
import graphviz 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
#
#
# --------------------- Print best parameters --------------------
#
#
def print_parameters(clf):
    print("Best parameters with score {:.5f}% set found on development set:".format(clf.best_score_))
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#
#
# --------------- Execute repeatitions of a hold out ---------------
#
#
def test_classifier(clf, repeats):
    global_predict = [list(),list()]
    for i in repeats:
        clf.fit(i[0], i[1])
        #print("----------------------------")
        #print(clf.estimators_)
        #print("----------------------------")
        #print(clf.estimators_samples_)
        #print("----------------------------")
        #print(clf.score(i[2],i[3]))
        #global_predict = clf.score(i[2],i[3])
        predict = list()
        for j in range(len(i[2])):
            a = np.argmax(np.squeeze(clf.predict_proba(np.array([i[2][j]]))))
            predict.append(a)
        correct = 0
        for j in range(len(predict)):
            if(predict[j] == i[3][j]):
                correct += 1
        global_predict[0].append(float(correct)/len(predict))
        global_predict[1].append(confusion_matrix(i[3],predict))
    return(np.array(global_predict[0]), global_predict[1])
#
#
# -------------- Print confusion matrix -----------
#
#
def print_confusion_matrix(confusions, which_matrix):
    if( len(confusions) < which_matrix ):
        return 0
    for i in range(len(confusions[which_matrix])):
        print("p\\a;", end="")
        for j in range(len(confusions[which_matrix][i])):
            print("{};".format(j), end="")
        break
    print()
    for i in range(len(confusions[which_matrix])):
        print("{};".format(i), end="")
        for j in range(len(confusions[which_matrix][i])):
            print("{};".format(confusions[which_matrix][i][j]), end="")
        print()
    return 1
#
# Combine results by vote
#
def CombineByVote(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                vote_list[np.argmax(np.array(i))] += 1
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
# Combine results by sum
#
def CombineBySum(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                for j in range(len(i)):
                    vote_list[j] += i[j]
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
# Combine results by product
#
def CombineByProduct(results):
    vote_list = [0 for i in range(len(results))]
    for i in range(len(results)):
        for j in len(range(results[i])):
            vote_list[j] *= results[i][j]
    return np.argmax(np.array(vote_list))
#
# Combine by max
#
def CombineByMax(results):
    return 0
#

#
#
#
# -------------- Start of the program ------------
#
#
if(len(sys.argv) != 2):
    print("Use: exercicio_1.py [base]")
    exit(0)
#
#
# ------------ Read the dataset ------------
#
#
base_file = open(sys.argv[1],"r")
#
base = arff.load(base_file)
#
# ------------ Load liver disorder dataset -------------
#
#
def load_liver_disorder(base):
    X = list()
    Y = list()
    T = list()
    #
    for i in base['data']:
        x_temp = list()
        x_temp.append(float(i[0]))
        x_temp.append(float(i[1]))
        x_temp.append(float(i[2]))
        x_temp.append(float(i[3]))
        x_temp.append(float(i[4]))
#        x_temp.append(float(i[5]))
        X.append(x_temp)
        if(float(i[5]) > 3):
            Y.append(1)
        else:
            Y.append(0)
        T.append(int(i[6]))
    X_train = list()
    X_test = list()
    Y_train = list()
    Y_test = list()
    for i in range(len(T)):
        if(T[i] == 1):
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return X,Y
#
#
# --------------- Load wine dataset
#
#
def load_wine(base):
    X = list()
    Y = list()
    #
    for i in base['data']:
        x_temp = list()
        for j in i[1:]:
            x_temp.append(float(j))
        X.append(x_temp)
        Y.append(int(i[0])-1)
    return X,Y
#
X, Y = load_wine(base)
#X, Y = load_liver_disorder(base)
#
#print(Y)
#print(len(X[0]))
base_file.close()
#
#
# ------------- Hold-out 70%-30% ----------
#
#
repeats = list()
#for i in range(10):
#    X_tmp_train, X_tmp_test, Y_tmp_train, Y_tmp_test = train_test_split(X, Y, test_size=0.30)
#    print(Y_tmp_test)
#    print("---------------------------")
#    repeats.append([X_tmp_train,Y_tmp_train,X_tmp_test,Y_tmp_test])
#
#
# ---------------- CV --------------------
#
#
fold_nr = 10
chunk_size = len(X)/fold_nr #print(round(float(len(X))/fold_nr))
X, Y = shuffle(X,Y)
for i in range(fold_nr):
    Y_tmp_train = list()
    Y_tmp_test = list()
    X_tmp_train = list()
    X_tmp_test = list()
    for j in range(chunk_size):
        idx = (i*chunk_size)+j
        if(idx < len(X)):
            Y_tmp_test.append(Y[(i*chunk_size)+j])
            X_tmp_test.append(X[(i*chunk_size)+j])
    if(i == 0):
        for j in range((i+1)*chunk_size,len(X)):
            Y_tmp_train.append(Y[j])
            X_tmp_train.append(X[j])
    else:
        for j in range(0,i*chunk_size):
            Y_tmp_train.append(Y[j])
            X_tmp_train.append(X[j])
        for j in range((i+1)*chunk_size,len(X)):
            Y_tmp_train.append(Y[j])
            X_tmp_train.append(X[j])
        print()
    repeats.append([X_tmp_train,Y_tmp_train,X_tmp_test,Y_tmp_test])
#
#
# -------------- Monolitic ---------------
#
##
#print("-----------------MLP-------------------")
##
#tuned_parameters = [{'hidden_layer_sizes': [(4), (6), (10), (20)], 'activation': ['identity', 'logistic', 'tanh', 'relu'],
#                    'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [1e-4, 1e-3, 1e-2], 
#                    'learning_rate': ['constant','invscaling','adaptive'],
#                    'max_iter': [100, 150, 200, 250],
#                    'tol': [1e-4, 1e-5, 1e-3],
#                    'momentum': [0.5, 0.7, 0.9]}]
##
##tuned_parameters = [{'hidden_layer_sizes': [(4), (6), (20)], 'activation': ['tanh', 'relu'],
##                    'solver': ['sgd'], 'alpha': [1e-4, 1e-3, 1e-2], 
##                    'learning_rate': ['adaptive'],
##                    'max_iter': [100, 200, 250],
##                    'tol': [1e-4, 1e-5, 1e-3],
##                    'momentum': [0.5, 0.9]}]
##
#mlp = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#scores, confusions = test_classifier(mlp, repeats)
#print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#print_confusion_matrix(confusions, scores.argmax())
#print_parameters(mlp)
##exit(0)
##
#print("-----------------Tree------------------")
##
#tuned_parameters = [{'splitter': ['best', 'random'], 'max_depth': [3,9,12],
#                    'max_leaf_nodes': [3,9,12], 'max_features': ['auto'], 
#                    'criterion': ['entropy']}]
#c45 = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#scores, confusions = test_classifier(c45, repeats)
#print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#print_confusion_matrix(confusions, scores.argmax())
#print_parameters(c45)
#monolithics.append(c45)
##
#print("-----------------Naive Bayes------------------")
##
#naive = GaussianNB()
#scores, confusions = test_classifier(naive, repeats)
#print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#print_confusion_matrix(confusions, scores.argmax())
##
#print("-----------------SVM------------------")
##
##tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
##                    'C': [5e-1, 5, 50, 500, 5000]},
##                    {'kernel': ['linear'], 'C': [1e-1, 1, 10, 100, 1000]}]
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2, 1, 2e-1],
#                    'C': [5e-1, 5, 10, 50, 500]},
#                     {'kernel': ['linear'], 'C': [1e-2, 1, 10, 100]}]
#smo = GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#scores, confusions = test_classifier(smo, repeats)
#print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#print_confusion_matrix(confusions, scores.argmax())
#print_parameters(smo)
##
#print("-----------------KNN------------------")
##
#tuned_parameters = [{'n_neighbors': [i for i in range(8,40)], 'leaf_size': [10, 30, 40, 50, 60, 70], 'metric': ['euclidean','minkowski']}]
##tuned_parameters = [{'n_neighbors': [i for i in range(1,30)], 'leaf_size': [10, 15, 30, 40, 50], 'metric': ['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean']}]
#neigh = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree', p=2), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4) #accuracy', n_jobs=4)
#scores, confusions = test_classifier(neigh, repeats)
#print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#print_confusion_matrix(confusions, scores.argmax())
#print_parameters(neigh)
#
#
# --------------- Bagging -----------------
#
##
#print("-----------------Bagging+CART (Classification and Regression Tree)------------------")
##
#c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best')
##c45 = tree.DecisionTreeClassifier()
#for i in [0.4, 0.5, 0.6, 0.7]:
#    for j in [10, 20, 30]:
#        print("Bagging: max_samples={} n_estimators={}".format(i,j))
#        bagging_bag = BaggingClassifier(c45, n_estimators=j, max_samples=i)
#        scores, confusions = test_classifier(bagging_bag, repeats)
#        print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
#        print_confusion_matrix(confusions, np.argmax(scores))
##
#print("-----------------RSS+CART------------------")
##
#c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best')
#for i in [0.5, 0.7, 0.9]:
#    for j in [5, 10, 25, 50, 75, 100]:
#        print("RSS: max_features={} n_estimators={}".format(i,j))
#        bagging_rss = BaggingClassifier(c45, n_estimators= j, max_samples=1, max_features=i)
#        scores, confusions = test_classifier(bagging_rss, repeats)
#        print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
#        print_confusion_matrix(confusions, scores.argmax())
##
#print("-----------------RandomForest------------------")
##
#for i in [5, 10, 25, 50, 75, 100]:
#    print("RandomForest: {}".format(i))
#    bagging_rf = RandomForestClassifier(n_estimators=i)
#    scores, confusions = test_classifier(bagging_rf, repeats)
#    print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#    print_confusion_matrix(confusions, scores.argmax())
##
#print("-----------------AdaBoost------------------")
##
#for i in [25, 50, 75, 100]:
#    print("Adaboost: {}".format(i))
#    boosting = AdaBoostClassifier(n_estimators=i)
#    scores, confusions = test_classifier(boosting, repeats)
#    print("Mean: {:.5f}, Std.Dev: {:.5f}, Best: {:.5f}, Worst: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
#    print_confusion_matrix(confusions, scores.argmax())
#
print("-----------SMC Heterogeneo ----------------")
#
#tuned_parameters = [{'hidden_layer_sizes': [(4), (6), (20)], 'activation': ['tanh', 'relu'],
#                    'solver': ['sgd'], 'alpha': [1e-4, 1e-3, 1e-2], 
#                    'learning_rate': ['adaptive'],
#                    'max_iter': [100, 200 ],
#                    'tol': [1e-5, 1e-1],
#                    'momentum': [0.5, 0.9]}]
tuned_parameters = [{'hidden_layer_sizes': [(4), (6), (20)], 'activation': ['tanh'],
                    'solver': ['adam'], 'alpha': [1e-3], 
                    'learning_rate': ['adaptive'],
                    'max_iter': [50, 100],
                    'tol': [1e-3],
                    'momentum': [0.5]}]
mlp = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#mlp = MLPClassifier()
#
tuned_parameters = [{'n_neighbors': [i for i in range(8,40)], 'leaf_size': [10, 30, 40, 50, 60, 70], 'metric': ['euclidean','minkowski']}]
neigh = GridSearchCV(KNeighborsClassifier(algorithm='kd_tree', p=2), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#neigh = KNeighborsClassifier()
#
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2, 1, 2e-1],
                    'C': [5e-1, 5, 50, 500]}]
#                     {'kernel': ['linear'], 'C': [1e-2, 1, 10, 100]}]
smo = GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#smo = svm.SVC(probability=True)
#
naive = GaussianNB()
#
tuned_parameters = [{'splitter': ['best', 'random'], 'max_depth': [3,9,12],
                    'max_leaf_nodes': [3,9,12], 'max_features': ['auto'], 
                    'criterion': ['entropy']}]
c45 = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
#c45 = tree.DecisionTreeClassifier()
#
preds = list()
for i in range(len(repeats)):
    c45.fit(repeats[i][0], repeats[i][1])
    naive.fit(repeats[i][0], repeats[i][1])
    smo.fit(repeats[i][0], repeats[i][1])
    neigh.fit(repeats[i][0], repeats[i][1])
    mlp.fit(repeats[i][0], repeats[i][1])
    correct = 0
    total = 0
    class_pred = list()
    for j in range(len(repeats[i][2])):
        results = list()
        results.append(np.squeeze(c45.predict_proba(np.array([repeats[i][2][j]]))))
        results.append(np.squeeze(naive.predict_proba(np.array([repeats[i][2][j]]))))
        results.append(np.squeeze(smo.predict_proba(np.array([repeats[i][2][j]]))))
        results.append(np.squeeze(neigh.predict_proba(np.array([repeats[i][2][j]]))))
        results.append(np.squeeze(mlp.predict_proba(np.array([repeats[i][2][j]]))))
        a = CombineByVote(results)
        class_pred.append(a)
        if(repeats[i][3][j] == a):
            correct += 1
        total += 1
    preds.append(float(correct)/total)
    print(float(correct)/total)
    print(confusion_matrix(repeats[i][3], class_pred))
np_preds = np.array(preds)
print("Mean: {} Stddev: {} Max: {} Min :{}".format(np.mean(np_preds), np.std(np_preds), np.max(np_preds), np.min(np_preds)))
exit(0)
#
#
#correct = 0
#total = 0
#for j in range(len(repeats)):
#    preds = list()
#    pred_class = list()
#    for i in monolithics:
#        print(repeats[j][0])
#        exit(0)
#        preds.append(np.squeeze(i.predict_proba(np.array(repeats[j][0]))))
#    if(CombineByVote(preds) == j[1]):
#        correct += 1
#    total += 1
#print("Acuracia ensemble heterogeneo: {}".format(float(correct)/total))
#
exit(0)
#
#dot_data = tree.export_graphviz(c45, out_file=None,
#                feature_names=['mean corpuscular volume', 'alkaline phosphotase', 'alamine aminotransferase', 'aspartate aminotransferase', 'gamma-glutamyl transpeptidase'],
#                class_names=['not drink','drink'],  
#                filled=True, rounded=True,  
#                special_characters=True)
#
#graph = graphviz.Source(dot_data) 
#graph.render("exercicio_liver_disorder_pdf") 
#clf.predict_proba([[2., 2.]])
#

