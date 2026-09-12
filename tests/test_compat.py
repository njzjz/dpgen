"""Regression tests for length checking on every supported Python version."""

import unittest

from dpgen._compat import zip_strict


class TestZipStrict(unittest.TestCase):
    def test_equal_length_inputs(self):
        self.assertEqual(list(zip_strict([1, 2], iter([3, 4]))), [(1, 3), (2, 4)])
        self.assertEqual(list(zip_strict()), [])
        self.assertEqual(list(zip_strict([], [])), [])
        self.assertEqual(list(zip_strict([None])), [(None,)])

    def test_mismatch_in_each_position(self):
        for lengths in ((0, 1), (1, 0), (2, 3, 2), (3, 2, 3)):
            with self.subTest(lengths=lengths):
                with self.assertRaisesRegex(ValueError, "different lengths"):
                    list(zip_strict(*(range(length) for length in lengths)))

    def test_consumption_is_lazy(self):
        consumed = []

        def source():
            for value in range(3):
                consumed.append(value)
                yield value

        pairs = zip_strict(source(), [10, 20])
        self.assertEqual(consumed, [])
        self.assertEqual(next(pairs), (0, 10))
        self.assertEqual(consumed, [0])
        self.assertEqual(next(pairs), (1, 20))
        with self.assertRaises(ValueError):
            next(pairs)

    def test_shared_iterator_detects_incomplete_group(self):
        values = iter(range(3))
        pairs = zip_strict(values, values)
        self.assertEqual(next(pairs), (0, 1))
        with self.assertRaises(ValueError):
            next(pairs)


if __name__ == "__main__":
    unittest.main()
