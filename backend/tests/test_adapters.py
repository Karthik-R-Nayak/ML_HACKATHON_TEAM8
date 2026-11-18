"""Isolated adapter tests use explicitly synthetic stage fixtures only."""
import unittest
from app.contracts import (Candidate, CandidateRequest, DecodeRequest, FeatureRequest,
                           ModelRequest, PreprocessRequest)
from app.fixtures import (FixtureCandidates, FixtureDecoder, FixtureFeatures,
                          FixtureModel, FixturePreprocessor)


class AdapterFixtureTests(unittest.TestCase):
    def setUp(self):
        self.pre = FixturePreprocessor().preprocess(PreprocessRequest(sequence="AC", alphabet="dna"))

    def test_preprocessing_fixture_preserves_positions(self):
        self.assertEqual(self.pre.positions, [0, 1])
        self.assertIn("fixture", self.pre.warnings[0].lower())

    def test_candidate_fixture_uses_valid_spans(self):
        result = FixtureCandidates().generate(CandidateRequest(sequence=self.pre.sequence, positions=self.pre.positions))
        self.assertEqual([(c.start, c.end) for c in result.candidates], [(0, 1), (1, 2)])

    def test_feature_fixture_has_stable_width(self):
        candidate = Candidate(candidate_id="fixture-c0", start=0, end=1, kind="fixture")
        result = FixtureFeatures().extract(FeatureRequest(sequence="A", candidates=[candidate]))
        self.assertEqual(len(result.vectors[0].values), len(result.feature_names))

    def test_model_and_decoder_fixtures_align(self):
        from app.contracts import FeatureResult, FeatureVector
        features = FeatureResult(feature_names=["fixture"], vectors=[FeatureVector(candidate_id=f"fixture-{i}", values=[0.0]) for i in range(2)])
        model = FixtureModel().predict(ModelRequest(features=features))
        decoded = FixtureDecoder().decode(DecodeRequest(scores=model.scores, labels=model.labels))
        self.assertEqual(len(decoded.path), 2)
        self.assertTrue(all(label.startswith("FIXTURE_") for label in decoded.path))


if __name__ == "__main__":
    unittest.main()
