import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn import metrics


def classification(latent_code, random_seed=42, ten_fold=False):
    tumour_type = pd.read_csv('data/PANCAN/GDC-PANCAN_both_samples_tumour_type.tsv', sep='\t', index_col=0)
    latent_code_label = pd.merge(latent_code, tumour_type, left_index=True, right_index=True)

    # separate latent variables and targets
    label = latent_code_label[['tumour_type']]
    data = latent_code_label.iloc[:, :-1]

    X = data.values
    y = label.values.ravel()

    if ten_fold:
        # 10-fold cross-validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
        accuracy_array = np.zeros(10)
        precision_array = np.zeros(10)
        recall_array = np.zeros(10)
        f1_array = np.zeros(10)
        i = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Use SVM as classifier
            clf = svm.SVC(gamma='scale', random_state=random_seed)
            clf.fit(X_train, y_train)

            # Test the classifier using the testing set
            y_pred = clf.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average='weighted')
            recall = metrics.recall_score(y_test, y_pred, average='weighted')
            f1 = metrics.f1_score(y_test, y_pred, average='weighted')

            # Store the metrics
            accuracy_array[i] = accuracy
            precision_array[i] = precision
            recall_array[i] = recall
            f1_array[i] = f1
            i = i + 1
        accuracy_average = np.mean(accuracy_array)
        precision_average = np.mean(precision_array)
        recall_average = np.mean(recall_array)
        f1_average = np.mean(f1_array)

        accuracy_std = accuracy_array.std()
        precision_std = precision_array.std()
        recall_std = recall_array.std()
        f1_std = f1_array.std()

        print('{:.2f}±{:.2f}%'.format(accuracy_average * 100, accuracy_std * 100))
        print('{:.3f}±{:.3f}'.format(precision_average, precision_std))
        print('{:.3f}±{:.3f}'.format(recall_average, recall_std))
        print('{:.3f}±{:.3f}'.format(f1_average, f1_std))

    else:
        testset_ratio = 0.2
        valset_ratio = 0.5

        # Just one separation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testset_ratio, random_state=random_seed,
                                                            stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=valset_ratio,
                                                        random_state=random_seed, stratify=y_test)

        # Use SVM as classifier
        clf = svm.SVC(gamma='scale', random_state=random_seed)
        clf.fit(X_train, y_train)

        # Test the classifier using the testing set
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')

        print('{:.2f}'.format(accuracy * 100))
        print('{:.2f}'.format(precision * 100))
        print('{:.2f}'.format(recall * 100))
        print('{:.2f}'.format(f1 * 100))
