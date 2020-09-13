import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cityblock, cosine
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings

def knn_classifier(X, y, iteration=100, neighbors_settings=range(1, 25)):
    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    df_precision = pd.DataFrame()
    df_recall = pd.DataFrame()
    
    for seedN in tqdm(range(0,iteration)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        stratify = y,
                                                        random_state=seedN)
        training_accuracy = []
        test_accuracy = []
        precision_value = []
        recall_value = []
        
        for n_neighbor in neighbors_settings:   
            clf = KNeighborsClassifier(n_neighbors=n_neighbor) 
            clf.fit(X_train, y_train) 
            training_accuracy.append(clf.score(X_train,y_train))
            test_accuracy.append(clf.score(X_test, y_test))
            
            y_predict = clf.predict(X_test)
            index = (np.where((y_predict == 'unstable'))[0])
            confusion_matrix = get_confusion("unstable", index, y_test)

            precision_value.append(precision(confusion_matrix))
            recall_value.append(recall(confusion_matrix))
            
        df_training[seedN] = training_accuracy
        df_test[seedN] = test_accuracy
        df_precision[seedN] = precision_value
        df_recall[seedN] = recall_value
    
    plt.errorbar(neighbors_settings, df_training.mean(axis=1),
                 yerr=df_training.var(axis=1), label="training accuracy", 
                 marker='o')
    plt.errorbar(neighbors_settings, df_test.mean(axis=1), marker='^',
                 yerr=df_test.var(axis=1), label="test accuracy")
    plt.errorbar(neighbors_settings, df_precision.mean(axis=1),
                 yerr=df_precision.var(axis=1), label="test precision", 
                 marker='o')
    plt.errorbar(neighbors_settings, df_recall.mean(axis=1), marker='^',
                 yerr=df_recall.var(axis=1), label="test recall")    
    
    plt.ylabel("Accuracy")
    plt.xlabel("kNN")
    plt.legend()
    
    print("Highest Test Recall = %f" % np.amax(df_recall.mean(axis=1)))
    print("Highest Test Precision = %f" % df_precision.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Training Accuracy = %f" % df_training.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Test Accuracy = %f" % df_test.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Best KNN Neighbor = %f" % neighbors_settings
                                      [np.argmax(df_recall.mean(axis=1))])


def classifier(classifier, X_scaled, y, iteration=100, **kwargs):
    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    df_precision = pd.DataFrame()
    df_recall = pd.DataFrame()
    C = [1e-3, 1e-2, 0.1, 0.2,0.4, 0.75, 1, 1.5, 3, 
         5, 10, 15,  20, 85, 100, 300, 1000]

    for seedN in tqdm(range(0,iteration)):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                           test_size=0.25, 
                                                           random_state=seedN,
                                                           stratify=y)
        training_accuracy = []
        test_accuracy = []
        precision_value = []
        recall_value = []
        
        for alpha_run in C:
            model = classifier(C=alpha_run, **kwargs).fit(X_train, y_train)
            training_accuracy.append(model.score(X_train, y_train))
            test_accuracy.append(model.score(X_test, y_test))

            y_predict = model.predict(X_test)
            index = (np.where((y_predict == 'unstable'))[0])
            confusion_matrix = get_confusion("unstable", index, y_test)

            precision_value.append(precision(confusion_matrix))
            recall_value.append(recall(confusion_matrix))

        df_training[seedN] = training_accuracy
        df_test[seedN] = test_accuracy
        df_precision[seedN] = precision_value
        df_recall[seedN] = recall_value

    plt.xscale('log')
    plt.errorbar(C, df_training.mean(axis=1),
                 yerr=df_training.var(axis=1), label="training accuracy", 
                 marker='o')
    plt.errorbar(C, df_test.mean(axis=1), marker='^',
                 yerr=df_test.var(axis=1), label="test accuracy")
    plt.errorbar(C, df_precision.mean(axis=1),
                 yerr=df_precision.var(axis=1), label="test precision", 
                 marker='o')
    plt.errorbar(C, df_recall.mean(axis=1), marker='^',
                 yerr=df_recall.var(axis=1), label="test recall")    
    
    plt.ylabel("Accuracy")
    plt.xlabel("C")
    plt.legend()
    print("Highest Test Recall = %f" % np.amax(df_recall.mean(axis=1)))
    print("Highest Test Precision = %f" % df_precision.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Training Accuracy = %f" % df_training.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Test Accuracy = %f" % df_test.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Best C Parameter = %f" % C[np.argmax(df_recall.mean(axis=1))])


def nsvm_classifier(X, y, iteration, kernel='poly'):

    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    df_precision = pd.DataFrame()
    df_recall = pd.DataFrame()
    C = [1000, 500, 100, 10, 5, 1]
    weighted_coefs=[]

    for seedN in tqdm(range(0,iteration)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=0.25, 
                                                           random_state=seedN,
                                                           stratify=y)
        training_accuracy = []
        test_accuracy = []
        precision_value = []
        recall_value = []
        C_parameter = []
        gamma_parameter = []

        for value in list(combinations_with_replacement(C, 2)):

            svm = SVC(kernel=kernel, degree=4, C = value[0], gamma=value[1])
            svc = svm.fit(X_train, y_train)
            training_accuracy.append(svc.score(X_train, y_train))
            test_accuracy.append(svc.score(X_test, y_test))
            C_parameter.append(value[0])
            gamma_parameter.append(value[1])

            y_predict = svc.predict(X_test)
            index = (np.where((y_predict == 'unstable'))[0])
            confusion_matrix = get_confusion("unstable", index, y_test)

            precision_value.append(precision(confusion_matrix))
            recall_value.append(recall(confusion_matrix))        

        df_training[seedN] = training_accuracy
        df_test[seedN] = test_accuracy
        df_precision[seedN] = precision_value
        df_recall[seedN] = recall_value

    print("Highest Test Recall = %f" % np.amax(df_recall.mean(axis=1)))
    print("Highest Test Precision = %f" % df_precision.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Training Accuracy = %f" % df_training.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Test Accuracy = %f" % df_test.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Best C Parameter = %f" % C_parameter
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Best gamma Parameter = %f" % gamma_parameter
                                      [np.argmax(df_recall.mean(axis=1))])
    

def tree_classifier(X_scaled, y, iteration=1, depths = 10):
    df_training = pd.DataFrame()
    df_test = pd.DataFrame()
    df_precision = pd.DataFrame()
    df_recall = pd.DataFrame()
    depth = range(1,depths)

    for seedN in tqdm(range(0,iteration)):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                           test_size=0.25, 
                                                           random_state=seedN,
                                                           stratify=y)
        training_accuracy = []
        test_accuracy = []
        precision_value = []
        recall_value = []
        
        for depth_run in depth:
            model = (DecisionTreeClassifier(criterion='gini', 
                                    max_depth=depth_run, 
                                    random_state=10).fit(X_train, y_train))
            training_accuracy.append(model.score(X_train, y_train))
            test_accuracy.append(model.score(X_test, y_test))

            y_predict = model.predict(X_test)
            index = (np.where((y_predict == 'unstable'))[0])
            confusion_matrix = get_confusion("unstable", index, y_test)

            precision_value.append(precision(confusion_matrix))
            recall_value.append(recall(confusion_matrix))

        df_training[seedN] = training_accuracy
        df_test[seedN] = test_accuracy
        df_precision[seedN] = precision_value
        df_recall[seedN] = recall_value
        
    plt.errorbar(depth, df_training.mean(axis=1),
                 yerr=df_training.var(axis=1), label="training accuracy", 
                 marker='o')
    plt.errorbar(depth, df_test.mean(axis=1), marker='^',
                 yerr=df_test.var(axis=1), label="test accuracy")
    plt.errorbar(depth, df_precision.mean(axis=1),
                 yerr=df_precision.var(axis=1), label="test precision", 
                 marker='o')
    plt.errorbar(depth, df_recall.mean(axis=1), marker='^',
                 yerr=df_recall.var(axis=1), label="test recall")    
    
    plt.ylabel("Accuracy")
    plt.xlabel("max_depth")
    plt.legend()
    print("Highest Test Recall = %f" % np.amax(df_recall.mean(axis=1)))
    print("Highest Test Precision = %f" % df_precision.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Training Accuracy = %f" % df_training.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Highest Test Accuracy = %f" % df_test.mean(axis=1)
                                      [np.argmax(df_recall.mean(axis=1))])
    print("Best Depth Parameter = %f" % depth
                                      [np.argmax(df_recall.mean(axis=1))])
    
def weight_analysis(best_classifier, X):
    fig = plt.figure(figsize=(8, 3))
    plt.plot(best_classifier.coef_.T, '--*')
    plt.xticks(range(len(X.columns)), X.columns, rotation=90)
    plt.hlines(0,0, len(X.columns))
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    return plt.show()

def top_predictor(best_classifier, X):
    df = pd.DataFrame(best_classifier.coef_, columns=X.columns).abs()
    best_weights = pd.concat([df.T.idxmax(), df.T.max()], axis=1)
    best_all = pd.DataFrame({0: [df.mean(0).idxmax()], 1: 
                      [df.mean(0).max()]}, index=['mean_all'])
    best_weights_all = pd.concat([best_weights, best_all])
    best_weights_all.columns = ['Top Predictor', 'Absolute Weight']
    display(best_weights_all)
        

def plot_feature_importances(model, dataframe):
    n_features=dataframe.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataframe.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
def get_confusion(actual, results, all_labels):
    all_labels = np.array(all_labels)
    result_labels = all_labels[results]
    labels = ['relevant', 'irrelevant']
   
    tp = sum(result_labels == actual)  
    tn = len(results) - tp   
    fp = sum(all_labels == actual) - tp   
    fn = sum(all_labels != actual) - tn   
    coef = np.array([[tp , tn], [fp, fn]])    
    return pd.DataFrame(coef, columns=labels, index=labels)

def nearest_k(query, objects, k, dist):
    return np.argsort([dist(query, obj) for obj in objects])[:k]

def precision(confusion):
    "Returns the precision of the confusion matrix"
    pres = confusion.loc['relevant'].relevant/confusion.loc['relevant'].sum()
    return pres if not np.isnan(pres) else 1.0

def recall(confusion):
    "Returns the recall of the confusion matrix"
    rec = confusion.loc['relevant'].relevant/confusion['relevant'].sum()
    return rec if not np.isnan(rec) else 1.0

def pr_curve(query, objects, dist, actual, all_labels):
    precision_list = [1]
    recall_list = [0]
    k_neighbors = nearest_k(query, objects, len(objects), dist)
    
    for k in range(1, len(objects) + 1):
        confusion = get_confusion(actual, k_neighbors[:k], all_labels)
        precision_list.append(precision(confusion))
        recall_list.append(recall(confusion))        

    fig, ax = plt.subplots()
    ax.plot(recall_list, precision_list)
    ax.fill_between(recall_list, precision_list, alpha=0.8)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    ax.text(0.65, 0.8,
        'AUC={:0.2f}'.format(np.trapz(precision_list,recall_list)),
         fontsize=12);
  
    return ax