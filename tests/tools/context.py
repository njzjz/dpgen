import os
import sys

from dpgen.tools.run_report import *

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def my_file_cmp(test, f0, f1):
    with open(f0) as fp0:
        with open(f1) as fp1:
            test.assertTrue(fp0.read() == fp1.read())
