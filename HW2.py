import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from warnings import filterwarnings


X, y = datasets.load_iris(return_X_y=True)

def adjust_labels_to_binary (y_train, target_class_value):
    np_array = y_train
    dict_irises = {'Setosa' : 0,'Versicolour' : 1, 'Virginica' : 2}
    np_array = np.where(np_array == dict_irises[target_class_value], 1, -1)
    return np_array



# Training a logistic regression model:
def one_vs_rest(x_train, y_train, target_class_value):
    y_train_binarized = adjust_labels_to_binary(y_train, target_class_value)
    log_reg = LogisticRegression()
    return log_reg.fit(x_train, y_train_binarized)

# Creating the confusion matrix from y and the prediction of y:
def binarized_confusion_matrix(X, y_binarized, one_vs_rest_model, prob_threshold):
    TP, FP, FN, TN = 0, 0, 0, 0 
    y_prediction = one_vs_rest_model.predict_proba(X)[:, 1]
    y_prediction_binarized = np.where(y_prediction >= prob_threshold, 1, -1)
    
    for i in range(len(y_binarized)):
        if y_prediction_binarized[i] == 1 and y_binarized[i] == 1:
            TP += 1
        if y_prediction_binarized[i] == 1 and y_binarized[i] == -1:
            FP += 1
        if y_prediction_binarized[i] == -1 and y_binarized[i] == 1:
            FN += 1
        if y_prediction_binarized[i] == -1 and y_binarized[i] == -1:
            TN += 1
    return np.array([[TP, FP], [FN, TN]])



# Printing the confusion matrices of Train and Test sets according to iris species:
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=98)

y_train_binarized_setosa = adjust_labels_to_binary(y_train, 'Setosa')
y_train_binarized_versicolour = adjust_labels_to_binary(y_train, 'Versicolour')
y_train_binarized_virginica = adjust_labels_to_binary(y_train, 'Virginica')

y_test_binarized_setosa = adjust_labels_to_binary(y_test, 'Setosa')
y_test_binarized_versicolour = adjust_labels_to_binary(y_test, 'Versicolour')
y_test_binarized_virginica = adjust_labels_to_binary(y_test, 'Virginica')

# Train confusion matrices:
print(binarized_confusion_matrix(X_train, y_train_binarized_setosa, one_vs_rest(X_train, y_train, 'Setosa'), 0.5))
print(binarized_confusion_matrix(X_train, y_train_binarized_versicolour, one_vs_rest(X_train, y_train, 'Versicolour'), 0.5))
print(binarized_confusion_matrix(X_train, y_train_binarized_virginica, one_vs_rest(X_train, y_train, 'Virginica'), 0.5))

# Test confusion matrices:
print(binarized_confusion_matrix(X_test, y_test_binarized_setosa, one_vs_rest(X_test, y_test, 'Setosa'), 0.5))
print(binarized_confusion_matrix(X_test, y_test_binarized_versicolour, one_vs_rest(X_test, y_test, 'Versicolour'), 0.5))
print(binarized_confusion_matrix(X_test, y_test_binarized_virginica, one_vs_rest(X_test, y_test, 'Virginica'), 0.5))



# Calculating the micro average precision on X, y and a prob_threshold given:
def micro_avg_precision(X, y, all_targed_class_dict, prob_threshold):
    # all_targed_class_dict['Setosa'] == one_vs_rest(X, y, 'Setosa')
    
    mat_setosa = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Setosa'), all_targed_class_dict['Setosa'], prob_threshold)
    mat_versicolour = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Versicolour'), all_targed_class_dict['Versicolour'], prob_threshold)
    mat_virginica = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Virginica'), all_targed_class_dict['Virginica'], prob_threshold)

    TP_sum = (mat_setosa[0,0] + mat_versicolour[0,0] + mat_virginica[0,0])
    FP_sum = (mat_setosa[0,1] + mat_versicolour[0,1] + mat_virginica[0,1])
    return (TP_sum / (TP_sum + FP_sum))


# Calculating the micro average recall on X, y and a prob_threshold given:
def micro_avg_recall(X, y, all_targed_class_dict, prob_threshold):

    mat_setosa = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Setosa'), all_targed_class_dict['Setosa'], prob_threshold)
    mat_versicolour = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Versicolour'), all_targed_class_dict['Versicolour'], prob_threshold)
    mat_virginica = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Virginica'), all_targed_class_dict['Virginica'], prob_threshold)

    TP_sum = (mat_setosa[0,0] + mat_versicolour[0,0] + mat_virginica[0,0])
    FN_sum = (mat_setosa[1,0] + mat_versicolour[1,0] + mat_virginica[1,0])
    return (TP_sum / (TP_sum + FN_sum))



# Calculating the micro average false positive rate on X, y and a prob_threshold given:
def micro_avg_false_positive_rate(X, y, all_targed_class_dict, prob_threshold):
    
    mat_setosa = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Setosa'), all_targed_class_dict['Setosa'], prob_threshold)
    mat_versicolour = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Versicolour'), all_targed_class_dict['Versicolour'], prob_threshold)
    mat_virginica = binarized_confusion_matrix(X, adjust_labels_to_binary(y, 'Virginica'), all_targed_class_dict['Virginica'], prob_threshold)

    TN_sum = (mat_setosa[1,1] + mat_versicolour[1,1] + mat_virginica[1,1])
    FP_sum = (mat_setosa[0,1] + mat_versicolour[0,1] + mat_virginica[0,1])
    return (FP_sum / (TN_sum + FP_sum))



# Plotting the ROC curve for the test set:
thresholds = [0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 1]
recall = []
fpr = []
all_targed_class_dict = {'Setosa': one_vs_rest(X_test, y_test, 'Setosa'), 'Versicolour': one_vs_rest(X_test, y_test, 'Versicolour'), 'Virginica': one_vs_rest(X_test, y_test, 'Virginica')}

for i in range(len(thresholds)):
    fpr.append(micro_avg_false_positive_rate(X_test, y_test, all_targed_class_dict, thresholds[i]))
    recall.append(micro_avg_recall(X_test, y_test, all_targed_class_dict, thresholds[i]))

plt.plot(fpr, recall)
plt.plot([0,0.2,0.4,0.6,1],[0,0.2,0.4,0.6,1], linestyle = 'dashed', color = 'green')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.title('ROC curve')
plt.xlabel('Recall')
plt.ylabel('False Positive Rate')
plt.grid(True)
fname=''
# plt.savefig(fname)
# plt.show()


# Calculating f_beta from the precision, recall and beta:
def f_beta(precision, recall, beta):
    f_beta = (1 + beta**2) * ((precision * recall) / (beta**2 * precision + recall))
    return f_beta



# Plotting f_beta as function of beta:

fig = plt.figure()
ax = plt.axes()

beta = np.linspace(0, 10, 1000)
ax.plot(beta, f_beta(micro_avg_precision(X_test, y_test, all_targed_class_dict, 0.3),micro_avg_recall(X_test, y_test, all_targed_class_dict,0.3),beta), label = 'prob thresh = 0.3')
ax.plot(beta, f_beta(micro_avg_precision(X_test, y_test, all_targed_class_dict, 0.5),micro_avg_recall(X_test, y_test, all_targed_class_dict,0.5),beta), label = 'prob thresh = 0.5')
ax.plot(beta, f_beta(micro_avg_precision(X_test, y_test, all_targed_class_dict, 0.7),micro_avg_recall(X_test, y_test, all_targed_class_dict,0.7),beta), label = 'prob thresh = 0.7')
plt.title('f_beta as function of beta')
plt.xlabel('beta')
plt.ylabel('f_beta')
plt.grid(True)
plt.xlim(0, 11)
plt.ylim(0.5, 1)
leg = ax.legend()
fname=''
# plt.savefig(fname)
# plt.show()




# Calculating a model's average train error and validation error:
def cross_validation_error(X, y, model, folds):
    avg_train_error = 0 
    avg_val_error = 0
    kf = KFold(n_splits=folds, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        fit_model = model.fit(X_train, y_train)
        X_train_pred = fit_model.predict(X_train)
        X_val_pred = fit_model.predict(X_val)
        avg_train_error += 1 - accuracy_score(X_train_pred, y_train)
        avg_val_error += 1 - accuracy_score(X_val_pred, y_val)
    return [avg_train_error / folds, avg_val_error / folds]


# Fitting a logistic regression model with different regularization and returning train, validation and test errors:
def Logistic_Regression_results(X_train, y_train, X_test, y_test):
    lambda_ = [10**(-4), 10**(-2), 1, 10**2, 10**4]
    train_error = [None]*5
    val_error = [None]*5
    test_error = [None]*5
    for i in range(len(lambda_)):
        [train_error[i], val_error[i]] = cross_validation_error(X_train, y_train, LogisticRegression(C = (1 / lambda_[i])), 5)
        fit_model = LogisticRegression(C = (1 / lambda_[i])).fit(X_train,y_train)
        test_pred = fit_model.predict(X_test)
        test_error[i] = 1 - accuracy_score(test_pred, y_test)

    return {'LogReg_lam_10^-4': [train_error[0], val_error[0], test_error[0]], 
            'LogReg_lam_10^-2': [train_error[1], val_error[1], test_error[1]], 
            'LogReg_lam_1':     [train_error[2], val_error[2], test_error[2]],
            'LogReg_lam_10^2':  [train_error[3], val_error[3], test_error[3]],
            'LogReg_lam_10^4':  [train_error[4], val_error[4], test_error[4]]}



# Loading the iris dataset and spliting it:
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=7)
filterwarnings('ignore') 



# Plotting the errors as function of lambda:
info = Logistic_Regression_results(X_train, y_train, X_test, y_test)

train_error = []
val_error = []
test_error = []
for i in info.keys():
    train_error.append(info[i][0])
    val_error.append(info[i][1])
    test_error.append(info[i][2])

labels = ("10^-4", "10^-2", "1", "10^2", "10^4")
X = np.arange(5)
fig, ax = plt.subplots()
ax.bar(X + 0.00, train_error, color = 'b', width = 0.25, label = 'Train Error')
ax.bar(X + 0.25, val_error, color = 'g', width = 0.25, label = 'Validation Error')
ax.bar(X + 0.50, test_error, color = 'r', width = 0.25, label = 'Test Error')
leg = ax.legend()
ax.set_title('Errors vs lambdas')
plt.xlabel('lambdas')
plt.ylabel('Errors')
ax.set_xticks(X)
ax.set_xticklabels(labels)
plt.grid(True)

for i in range(5):
    plt.text(i-0.20 , train_error[i] + 0.01, color ='b', s=str(np.around(train_error[i], decimals=2)))
    plt.text(i+0.10 , val_error[i] + 0.01, color = 'g', s=str(np.around(val_error[i], decimals=2)))
    plt.text(i+0.45, test_error[i] + 0.01, color = 'r', s=str(np.around(test_error[i], decimals=2)))

fname=''
# plt.savefig(fname)
plt.show()
