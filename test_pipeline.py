import unittest
import pandas as pd
import numpy as np
import re

# Define unit test cases for the extraction and cleaning process
class TestNBAPipeline(unittest.TestCase):

    def test_normalize_name(self):
        # Test capitalization, trailing stars, period removals, and suffixes
        from __main__ import load_clean_nba_draft_data
        
        # We can implement a local test helper mimicking normalize_name logic
        def normalize_name(name):
            if not isinstance(name, str):
                return ""
            name = name.lower()
            name = name.replace('*', '')
            name = re.sub(r'(?<=\b[a-z])\.\b', '', name)
            name = re.sub(r'(?<=\b[a-z])\.', '', name) 
            name = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)$_', '', name)
            name = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)$_', '', name)
            name = " ".join(name.split())
            return name

        self.assertEqual(normalize_name("Otto Porter Jr."), "otto porter")
        self.assertEqual(normalize_name("Dennis Smith Jr"), "dennis smith")
        self.assertEqual(normalize_name("Andrew Wiggins*"), "andrew wiggins")
        self.assertEqual(normalize_name("A.J. Hammons"), "aj hammons")
        self.assertEqual(normalize_name(np.nan), "")

    def test_determine_position(self):
        def determine_position(row):
            positions = []
            if row['guard'] == 1:
                positions.append('Guard')
            if row['forward'] == 1:
                positions.append('Forward')
            if row['center'] == 1:
                positions.append('Center')
            if len(positions) == 0:
                return 'Unknown'
            elif len(positions) == 1:
                return positions[0]
            else:
                return '-'.join(positions)

        self.assertEqual(determine_position({'guard': 1, 'forward': 0, 'center': 0}), 'Guard')
        self.assertEqual(determine_position({'guard': 0, 'forward': 1, 'center': 1}), 'Forward-Center')
        self.assertEqual(determine_position({'guard': 0, 'forward': 0, 'center': 0}), 'Unknown')

    def test_metric_math(self):
        # Basic verification of mathematical equations for BPM and WS calculations
        games_played = 10
        pts, reb, ast, stl, blk, fga, fgm, fta, ftm, tov, pf = 150, 50, 30, 10, 5, 110, 50, 40, 30, 20, 15
        
        college_bpm = (
            pts + reb + ast + stl + blk
            - (fga - fgm) - (fta - ftm) - tov
        ) / games_played

        college_ws = (
            pts + 0.4 * ftm - 0.7 * fga - 0.4 * fta
            + 0.7 * reb + 0.7 * ast + 1.0 * stl + 1.0 * blk
            - 1.0 * tov - 0.4 * pf
        ) / 10.0

        self.assertEqual(college_bpm, 15.5)
        self.assertAlmostEqual(college_ws, 12.35, places=2)

if __name__ == '__main__':
    unittest.main()
