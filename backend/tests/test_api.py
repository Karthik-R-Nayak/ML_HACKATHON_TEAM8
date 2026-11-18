"""API integration tests use synthetic fixture adapters explicitly."""
import os
import unittest
from fastapi.testclient import TestClient
from app.main import create_app
from app.fixtures import fixture_pipeline


class ApiTests(unittest.TestCase):
    def test_unconfigured_pipeline_is_not_a_prediction(self):
        client = TestClient(create_app())
        response = client.post("/v1/analyze", json={"sequence": "ACGT"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "adapter_not_configured")

    def test_fixture_pipeline_is_labelled(self):
        client = TestClient(create_app(pipeline=fixture_pipeline()))
        response = client.post("/v1/analyze", json={"sequence": "ACGT", "sequence_id": "test-fixture"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["is_fixture"])
        self.assertEqual(body["notice"], "TEST FIXTURE — NOT A PREDICTION")
        self.assertTrue(body["model_name"].startswith("synthetic-"))

    def test_invalid_request_is_rejected(self):
        client = TestClient(create_app(pipeline=fixture_pipeline()))
        self.assertEqual(client.post("/v1/analyze", json={"sequence": "  "}).status_code, 422)

    def test_fixtures_forbidden_in_production(self):
        old_env, old_fixture = os.environ.get("APP_ENV"), os.environ.get("ENABLE_TEST_FIXTURES")
        try:
            os.environ["APP_ENV"] = "production"
            os.environ["ENABLE_TEST_FIXTURES"] = "true"
            with self.assertRaises(RuntimeError):
                create_app()
        finally:
            if old_env is None: os.environ.pop("APP_ENV", None)
            else: os.environ["APP_ENV"] = old_env
            if old_fixture is None: os.environ.pop("ENABLE_TEST_FIXTURES", None)
            else: os.environ["ENABLE_TEST_FIXTURES"] = old_fixture


if __name__ == "__main__":
    unittest.main()
