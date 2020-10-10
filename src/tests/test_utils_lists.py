# Created 17 June 2020

import unittest

from src.utils_lists import flatten_list, has_duplicates, list_as_comma_sep, \
    all_unique, chunk, count_occurrences, flatten


class TestUtilsLists(unittest.TestCase):
    def test_flatten_list(self):
        self.assertEqual(flatten_list([1, 2, 3, [4, 5]]),
                         [1, 2, 3, 4, 5],
                         "Resulting list should be [1,2,3,4,5]")

    def test_has_duplicates__true(self):
        self.assertEqual(has_duplicates(lst=[1, 2, 3, 4, 4]),
                         True,
                         "Value 4 is repeated")

    def test_has_duplicates__false(self):
        self.assertEqual(has_duplicates(lst=[1, 2, 3, 4, 5]),
                         False,
                         "No repeated values")

    def test_list_as_comma_sep(self):
        self.assertEqual(list_as_comma_sep(lst=['hello', 'test', 'list']),
                         'hello,test,list',
                         "Should've returned 'hello,test,list' ")

    def test_all_unique(self):
        self.assertEqual(all_unique(lst=[1, 2, 3, 4]),
                         True,
                         "All elements are unique")

    def test_chunk(self):
        self.assertEqual(chunk(lst=[1, 2, 3, 4, 5, 6], chunk_size=2),
                         [[1, 2], [3, 4], [5, 6]],
                         "Resulting list should be [[1, 2], [3, 4], [5, 6]]")

    def test_count_occurences(self):
        self.assertEqual(count_occurrences(lst=[1, 2, 3, 4, 2, 2, 2, 2], value=2),
                         5,
                         "THe number 2 appears 5 times")

    def test_flatten(self):
        self.assertEqual(flatten(lst=[1, 2, [3, 4, 5, [6, 7]]]),
                         [1, 2, 3, 4, 5, 6, 7],
                         "Flattened list should be [1, 2, 3, 4, 5, 6, 7]")


if __name__ == '__main__':
    unittest.main()
