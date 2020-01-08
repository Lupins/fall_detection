import sys
from sklearn import metrics

def main(TP, FP, FN, TN):

    y_true = []
    y_pred = []

    for i in range(TP):
        y_true.append(0)
        y_pred.append(0)

    for i in range(FP):
        y_true.append(1)
        y_pred.append(0)

    for i in range(FN):
        y_true.append(0)
        y_pred.append(1)

    for i in range(TN):
        y_true.append(1)
        y_pred.append(1)

    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred))
    print('Balanced Accuracy:\t', '{0:.2f}'.format(metrics.balanced_accuracy_score(y_true, y_pred)))
    print('Matthews:\t\t', '{0:.2f}'.format(metrics.matthews_corrcoef(y_true, y_pred)))

TP = int(sys.argv[1])
FP = int(sys.argv[2])
FN = int(sys.argv[3])
TN = int(sys.argv[4])

main(TP, FP, FN, TN)
