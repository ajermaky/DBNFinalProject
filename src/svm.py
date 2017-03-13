import timeit

from src.data import load_data
from sklearn import svm, metrics
import sys


def run_svm(digits=None, gamma=.001, C=1, fd=sys.stdout):
    if digits is None:
        digits = load_data('mnist.pkl.gz')

    train_set_x, train_set_y = digits[0]
    test_set_x, test_set_y = digits[2]
    classifier = svm.SVC(gamma=gamma, C=C)

    fd.write("fitting\n")
    start = timeit.default_timer()
    classifier.fit(train_set_x, train_set_y)
    end = timeit.default_timer()
    fd.write("fitting completed. Took  %.2fm\n" % ((end - start) / 60.))
    fd.write("predicting\n")
    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(test_set_x)
    fd.write('prediction complete\n')

    fd.write("done\n\n")
    fd.write("Classification report for classifier %s:\n%s\n"
             % (classifier, metrics.classification_report(test_set_y, predicted)))
    fd.write("Accuracy: %s\n" % metrics.accuracy_score(test_set_y, predicted))
    fd.write("Confusion matrix:\n%s" % metrics.confusion_matrix(test_set_y, predicted))


if __name__ == '__main__':
    digits = load_data('mnist.pkl.gz')
    fd = open("hello.log", 'w+')
    train_set_x, train_set_y = digits[0]
    train_set_x = train_set_x[:1000]
    train_set_y = train_set_y[:1000]

    digits[0] = (train_set_x, train_set_y)
    run_svm(digits=digits, C=1, gamma=.001, fd=fd)
    fd.close()
