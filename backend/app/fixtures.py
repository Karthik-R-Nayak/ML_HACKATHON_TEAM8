"""Explicitly synthetic integration fixtures. Never enable for production inference."""
from app.contracts import *
from app.pipeline import Pipeline


class FixturePreprocessor:
    name, version = "synthetic-preprocessor", "fixture-v1"
    def preprocess(self, request):
        return PreprocessResult(sequence=request.sequence, positions=list(range(len(request.sequence))),
                               warnings=["Synthetic integration fixture; not scientific preprocessing."])


class FixtureCandidates:
    name, version = "synthetic-candidates", "fixture-v1"
    def generate(self, request):
        return CandidateResult(candidates=[Candidate(candidate_id=f"fixture-{i}", start=i, end=i+1, kind="fixture-symbol") for i in range(len(request.sequence))])


class FixtureFeatures:
    name, version = "synthetic-features", "fixture-v1"
    def extract(self, request):
        return FeatureResult(feature_names=["fixture_constant"], vectors=[FeatureVector(candidate_id=c.candidate_id, values=[0.0]) for c in request.candidates])


class FixtureModel:
    name, version = "synthetic-model", "fixture-v1"
    def predict(self, request):
        count = len(request.features.vectors)
        return ModelResult(model_name=self.name, model_version=self.version, labels=["FIXTURE_A", "FIXTURE_B"], scores=[[0.5, 0.5] for _ in range(count)])


class FixtureDecoder:
    name, version = "synthetic-viterbi", "fixture-v1"
    def decode(self, request):
        return DecodeResult(path=[request.labels[0]] * len(request.scores), log_probability=0.0)


def fixture_pipeline() -> Pipeline:
    return Pipeline(FixturePreprocessor(), FixtureCandidates(), FixtureFeatures(), FixtureModel(), FixtureDecoder(), fixture=True)
