"""PathSnap integration tests use small explicitly synthetic network fixtures."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.adapters.candidate_generation import PathSnapCandidateGenerationAdapter
from app.adapters.feature_extraction import PathSnapFeatureExtractionAdapter
from app.adapters.preprocessing import PathSnapPreprocessingAdapter
from app.contracts import PathSnapPreprocessRequest
from app.main import create_app
from app.pathsnap.pipeline import PathSnapPreprocessingPipeline
from app.pathsnap.preprocessing.config import CandidateConfig, FeatureConfig, IOConfig
from app.pathsnap.preprocessing.dataset import process_trip, run_training_pipeline, training_matrix
from app.pathsnap.preprocessing.features import FEATURE_COLUMNS, observation_features
from app.pathsnap.preprocessing.labels import recall_report
from app.pathsnap.splits import discover_dataset_splits


MODULES = (
    "labels", "align", "dataset", "config", "candidates", "features", "io", "geo", "index", "network",
)


def small_network_fixture():
    """Return labelled synthetic coordinates; not real map data or a prediction fixture."""
    nodes = pd.DataFrame({"lon": [0.0, 0.001, 0.002], "lat": [0.0, 0.0, 0.0]})
    arcs = pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]})
    track = pd.DataFrame({
        "lon": [0.0001, 0.0008, 0.0014, 0.0019],
        "lat": [0.00005, 0.00003, 0.00006, 0.00002],
        "t": [1.0, 2.0, 3.0, 4.0],
    })
    return nodes, arcs, track


class PathSnapIntegrationTests(unittest.TestCase):
    def test_all_ten_imported_modules_import(self):
        for module_name in MODULES:
            with self.subTest(module=module_name):
                imported = importlib.import_module(f"app.pathsnap.preprocessing.{module_name}")
                self.assertIsNotNone(imported)
        import io as standard_io
        self.assertTrue(hasattr(standard_io, "BytesIO"))

    def test_network_construction_and_original_arc_ids(self):
        nodes, arcs, track = small_network_fixture()
        net, xy, timestamps = PathSnapPreprocessingAdapter().prepare(
            nodes, arcs, track, IOConfig(node_id_base=0, arc_id_base=100),
        )
        self.assertEqual(net.n_arcs, 2)
        self.assertEqual(net.arc_ids.tolist(), [100, 101])
        self.assertEqual(xy.shape, (4, 2))
        self.assertEqual(timestamps.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_candidate_generation_and_feature_column_order(self):
        nodes, arcs, track = small_network_fixture()
        prep = PathSnapPreprocessingAdapter()
        net, xy, timestamps = prep.prepare(nodes, arcs, track, IOConfig())
        _, candidates, n_within = PathSnapCandidateGenerationAdapter().generate_for_track(net, xy, CandidateConfig())
        features, observations, columns = PathSnapFeatureExtractionAdapter().extract_for_track(
            net, candidates, xy, timestamps, FeatureConfig(heading_mode="central"),
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(columns, FEATURE_COLUMNS)
        self.assertEqual(features["arc_idx"].min(), 0)
        self.assertTrue((features["arc_id"] == features["arc_idx"]).all())
        self.assertEqual(len(n_within), len(track))
        self.assertEqual(len(observations), len(track))

    def test_training_matrix_excludes_candidate_miss_groups_but_report_keeps_them(self):
        rows = pd.DataFrame([
            {"trip_id": "train-01", "obs_idx": 0, "label_valid": True, "label": 1, "dist_m": 1.0},
            {"trip_id": "train-01", "obs_idx": 0, "label_valid": True, "label": 0, "dist_m": 2.0},
            {"trip_id": "train-01", "obs_idx": 1, "label_valid": True, "label": 0, "dist_m": 3.0},
            {"trip_id": "train-02", "obs_idx": 0, "label_valid": False, "label": 0, "dist_m": 4.0},
        ])
        for column in FEATURE_COLUMNS:
            if column not in rows:
                rows[column] = 0.0
        x, y, groups = training_matrix(rows)
        self.assertEqual(len(x), 2)
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(groups.tolist(), ["train-01", "train-01"])

        observations = pd.DataFrame({
            "status": ["aligned", "aligned"],
            "gt_in_candidates": [True, False],
            "miss_reason": ["", "radius"],
            "n_candidates": [2, 0],
            "gt_rank": [1.0, np.nan],
        })
        report = recall_report(observations)
        self.assertEqual(report["n_labeled"], 2)
        self.assertEqual(report["miss_radius"], 1)
        self.assertEqual(report["miss_rate"], 0.5)

    def test_dataset_splits_use_existing_directories_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, trip_id in (("train", "00000001"), ("train", "00000002"),
                                   ("student_test", "00000000")):
                trip_dir = root / split / trip_id
                trip_dir.mkdir(parents=True)
                for suffix in ("track", "nodes", "arcs"):
                    (trip_dir / f"{trip_id}.{suffix}").write_text("synthetic fixture\n")
            result = discover_dataset_splits(root)
            self.assertEqual([trip[0] for trip in result.train], ["00000001", "00000002"])
            self.assertEqual([trip[0] for trip in result.student_test], ["00000000"])
            self.assertIsNone(result.evaluator_test)

            eval_dir = root / "evaluator_test" / "00000999"
            eval_dir.mkdir(parents=True)
            for suffix in ("track", "nodes", "arcs"):
                (eval_dir / f"00000999.{suffix}").write_text("synthetic fixture\n")
            result = discover_dataset_splits(root)
            self.assertEqual([trip[0] for trip in result.evaluator_test], ["00000999"])
            self.assertNotIn("00000999", {trip[0] for trip in result.train})

    def test_training_pipeline_targets_train_partition_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp) / "train"
            train_dir.mkdir()
            (Path(tmp) / "student_test").mkdir()
            (Path(tmp) / "evaluator_test").mkdir()
            with patch("app.pathsnap.preprocessing.dataset.run_pipeline", return_value=(None, None, {})) as run:
                run_training_pipeline(tmp)
            self.assertEqual(run.call_args.args[0], train_dir)

    def test_central_heading_and_route_independent_inference(self):
        xy = np.array([[0.0, 0.0], [5.0, 0.0], [15.0, 0.0]])
        times = np.array([0.0, 1.0, 3.0])
        obs = observation_features(xy, times, FeatureConfig(heading_mode="central"))
        self.assertEqual(obs.loc[1, "gps_bearing_deg"], 90.0)

        nodes, arcs, track = small_network_fixture()
        output = PathSnapPreprocessingPipeline().run(
            trip_id="fixture-route-free", nodes=nodes, arcs=arcs, track=track,
            node_id_base=0, arc_id_base=100, route_arc_ids=None,
        )
        self.assertFalse(output["labeling_performed"])
        self.assertIsNone(output["route_arc_ids"])
        self.assertFalse(output["is_prediction"])
        self.assertEqual(output["feature_config"]["heading_mode"], "central")
        self.assertTrue(output["candidates"])
        self.assertNotIn("label", output["candidates"][0])
        self.assertIn("arc_id", output["candidates"][0])
        self.assertIn("arc_idx", output["candidates"][0])
        self.assertAlmostEqual(output["observations"][0]["gps_lon"], float(track.iloc[0]["lon"]))
        self.assertAlmostEqual(output["observations"][0]["gps_lat"], float(track.iloc[0]["lat"]))

    def test_dataset_processing_rejects_noncentral_training_config(self):
        from app.pathsnap.preprocessing.candidates import build_index
        from app.pathsnap.preprocessing.config import PipelineConfig
        from app.pathsnap.preprocessing.network import build_network
        nodes, arcs, track = small_network_fixture()
        net = build_network(nodes, arcs)
        index = build_index(net, CandidateConfig())
        bad_cfg = PipelineConfig(feat=FeatureConfig(heading_mode="backward"))
        with self.assertRaisesRegex(ValueError, "heading_mode='central'"):
            process_trip("fixture", net, index, track, route_arc_ids=None, cfg=bad_cfg)

    def test_fastapi_calls_geospatial_preprocessing_without_route(self):
        nodes, arcs, track = small_network_fixture()
        payload = {
            "trip_id": "fixture-api-trip",
            "observations": track.to_dict(orient="records"),
            "nodes": nodes.to_dict(orient="records"),
            "arcs": arcs.to_dict(orient="records"),
            "node_id_base": 0,
            "arc_id_base": 100,
            "route_arc_ids": None,
        }
        response = TestClient(create_app()).post("/v1/pathsnap/preprocess", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["trip_id"], "fixture-api-trip")
        self.assertFalse(body["labeling_performed"])
        self.assertFalse(body["is_prediction"])
        self.assertEqual(body["network"]["arc_id_base"], 100)
        self.assertIn("arc_id", body["candidates"][0])
        self.assertIn("arc_idx", body["candidates"][0])
        self.assertIn("obs_idx", body["candidates"][0])
        self.assertEqual(body["candidate_feature_columns"], FEATURE_COLUMNS)
        self.assertEqual(body["observations"][0]["gps_lon"], track.iloc[0]["lon"])
        self.assertEqual(body["observations"][0]["gps_lat"], track.iloc[0]["lat"])
        self.assertIn("obs_idx", body["observation_feature_columns"])

    def test_invalid_network_reference_is_http_422(self):
        nodes, arcs, track = small_network_fixture()
        arcs.loc[0, "to_node"] = 99
        payload = {
            "trip_id": "fixture-invalid-map", "observations": track.to_dict(orient="records"),
            "nodes": nodes.to_dict(orient="records"), "arcs": arcs.to_dict(orient="records"),
        }
        response = TestClient(create_app()).post("/v1/pathsnap/preprocess", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_pathsnap_input")


if __name__ == "__main__":
    unittest.main()
