import os
import sys

class ConditionalPrint:
    def __init__(self, enable_print):
        self._enable_print = enable_print

    def __enter__(self):
        if not self._enable_print:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enable_print:
            sys.stdout.close()
            sys.stdout = self._original_stdout