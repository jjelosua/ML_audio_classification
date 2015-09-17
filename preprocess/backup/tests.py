import unittest
import numpy as np
from featurize import extract_chunks


class FeaturizerTests(unittest.TestCase):
    def test_extract_chunks(self):
        data = np.array([0,0,0,0,0,1,4,6,8,1,0,0,0,0,2,2,0])
        silences, nonsilences = extract_chunks(data)

        expected_silences = [[0, 5], [10, 4], [16, 1]]
        expected_nonsilences = [(5, [1,4,6,8,1]), (14, [2,2])]

        self.assertEqual(silences, expected_silences)
        self.assertEqual(nonsilences, expected_nonsilences)

        data = np.array([1,4,6,8,1,0,0,0,0,2,2,0])
        silences, nonsilences = extract_chunks(data)

        expected_silences = [[5, 4], [11, 1]]
        expected_nonsilences = [(0, [1,4,6,8,1]), (9, [2,2])]

        self.assertEqual(silences, expected_silences)
        self.assertEqual(nonsilences, expected_nonsilences)

    def test_preceding_silence_lengths(self):
        nonsilences = [(5, [1,4,6,8,1]), (34, [2,2])]
        silences = [[0,5], [14, 20], [36, 13]]
        preceding_silence_lengths = [next((length for spos, length in silences[::-1] if spos < pos), 0) for pos, ns in nonsilences]
        self.assertEqual([5, 20], preceding_silence_lengths)
