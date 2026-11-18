"""Orchestrates validated adapters without embedding scientific algorithms."""
from dataclasses import dataclass
from math import isfinite
from app.contracts import (AnalysisResponse, CandidateRequest, DecodeRequest,
                           FeatureRequest, ModelRequest, PreprocessRequest, SequenceInput)
from app.adapters.preprocessing import Preprocessor
from app.adapters.candidate_generation import CandidateGenerator
from app.adapters.feature_extraction import FeatureExtractor
from app.adapters.ml_model import SequenceModel
from app.adapters.hmm_viterbi import SequenceDecoder


class AdapterNotConfigured(RuntimeError):
    pass


@dataclass
class Pipeline:
    preprocessor: Preprocessor | None = None
    candidate_generator: CandidateGenerator | None = None
    feature_extractor: FeatureExtractor | None = None
    model: SequenceModel | None = None
    decoder: SequenceDecoder | None = None
    fixture: bool = False

    def capabilities(self) -> dict:
        stages = {
            "preprocessing": self.preprocessor, "candidate_generation": self.candidate_generator,
            "feature_extraction": self.feature_extractor, "ml_model": self.model,
            "hmm_viterbi": self.decoder,
        }
        return {name: ({"ready": True, "name": getattr(adapter, "name", "unknown"),
                        "version": getattr(adapter, "version", "unknown")} if adapter else {"ready": False})
                for name, adapter in stages.items()} | {"pipeline_ready": all(stages.values()), "fixture_mode": self.fixture}

    def analyze(self, request: SequenceInput) -> AnalysisResponse:
        adapters = (self.preprocessor, self.candidate_generator, self.feature_extractor, self.model, self.decoder)
        if not all(adapters):
            missing = [name for name, value in zip(("preprocessing", "candidate_generation", "feature_extraction", "ml_model", "hmm_viterbi"), adapters) if value is None]
            raise AdapterNotConfigured("Adapters not configured: " + ", ".join(missing))
        pre = self.preprocessor.preprocess(PreprocessRequest(sequence=request.sequence, alphabet=request.alphabet))
        if any(position < 0 or position >= len(request.sequence) for position in pre.positions):
            raise ValueError("preprocessor positions must reference symbols in the submitted sequence")
        cand = self.candidate_generator.generate(CandidateRequest(sequence=pre.sequence, positions=pre.positions))
        if any(item.end > len(pre.sequence) for item in cand.candidates):
            raise ValueError("candidate span exceeds normalized sequence length")
        if len({item.candidate_id for item in cand.candidates}) != len(cand.candidates):
            raise ValueError("candidate IDs must be unique")
        feats = self.feature_extractor.extract(FeatureRequest(sequence=pre.sequence, candidates=cand.candidates))
        if [vector.candidate_id for vector in feats.vectors] != [item.candidate_id for item in cand.candidates]:
            raise ValueError("feature vectors must correspond one-to-one and in order with candidates")
        model = self.model.predict(ModelRequest(features=feats))
        if len(model.scores) != len(pre.sequence):
            raise ValueError("model must return one score row per normalized sequence symbol")
        decoded = self.decoder.decode(DecodeRequest(scores=model.scores, labels=model.labels))
        if len(decoded.path) != len(pre.sequence) or any(label not in model.labels for label in decoded.path):
            raise ValueError("decoder path must contain one valid label per sequence symbol")
        if not isfinite(decoded.log_probability):
            raise ValueError("decoder log_probability must be finite")
        return AnalysisResponse(sequence_id=request.sequence_id, sequence_length=len(pre.sequence),
            normalized_sequence=pre.sequence, candidates=cand.candidates,
            feature_names=feats.feature_names, model_name=model.model_name, model_version=model.model_version,
            labels=model.labels, scores=model.scores, decoded_path=decoded.path,
            warnings=pre.warnings, is_fixture=self.fixture,
            notice="TEST FIXTURE — NOT A PREDICTION" if self.fixture else None)
