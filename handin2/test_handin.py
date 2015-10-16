import os
import sys
import argparse
import importlib
import contextlib
import load_data

import numpy as np


@contextlib.contextmanager
def redirect_print(filename):
    """
    Redirects "print" to a file.

    >>> with redirect_print('test.txt'):
    ...     print("This goes into the file")
    >>> open('test.txt').read()
    'This goes into the file\n'
    """

    with open(filename, 'w') as fp:
        stdout, sys.stdout = sys.stdout, fp
        stderr, sys.stderr = sys.stderr, fp
        try:
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    base, ext = os.path.splitext(args.filename)
    if ext != '.py':
        parser.error("Filename must end in .py")
    dirname, basename = os.path.split(base)
    sys.path.insert(0, dirname)
    m = importlib.import_module(basename)

    # Let's see how well the handin performs on these digits
    n = 30
    digits = np.random.uniform(0, 1, (n, 784))
    random_labels = np.repeat(9, n)

    print_file = 'handin-output.txt'
    with redirect_print(print_file):
        prediction = m.predict(digits)
    prediction_type = type(prediction).__name__
    prediction = np.asarray(prediction)
    print("predict() returned type=%s; dtype=%s; shape=%s" %
          (prediction_type, prediction.dtype, prediction.shape))
    correct = (random_labels == prediction.ravel())
    print("Correct: %d  Incorrect: %d" % (correct.sum(), (~correct).sum()))
    with open(print_file) as fp:
        line_count = sum(1 for line in fp)
    print("%d line%s in %s" % (line_count, '' if line_count == 1 else 's',
                               print_file))


if __name__ == "__main__":
    main()
