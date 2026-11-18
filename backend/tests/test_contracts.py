"""Contract-only tests; every datum is a synthetic fixture, not scientific output."""
import unittest
from pydantic import ValidationError
from app.contracts import Candidate, FeatureResult, FeatureVector, SequenceInput


class ContractTests(unittest.TestCase):
    def test_sequence_normalization(self):
        self.assertEqual(SequenceInput(sequence=" ac\ngt ").sequence, "ACGT")

    def test_rejects_blank_sequence(self):
        with self.assertRaises(ValidationError):
            SequenceInput(sequence="  \n")

    def test_candidate_uses_half_open_nonempty_range(self):
        with self.assertRaises(ValidationError):
            Candidate(candidate_id="fixture", start=2, end=2, kind="fixture")

    def test_feature_dimensions(self):
        with self.assertRaises(ValidationError):
            FeatureResult(feature_names=["fixture_a", "fixture_b"], vectors=[FeatureVector(candidate_id="fixture", values=[0.0])])


if __name__ == "__main__":
    unittest.main()
