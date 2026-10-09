"""Test configuration.

When FAKE_MODEL=1 is set (as in CI), the app's startup model load is
replaced with a tiny stand-in model. CI has no access to the MLflow
registry on EC2/S3, so the real model cannot be loaded there; the
stand-in mimics the real model's interface (predict returns class
indices 0/1/2) so the API routes, validation and response shape are
still genuinely tested. Without the flag, tests use the real model.
"""
import os

import pytest

if os.getenv("FAKE_MODEL"):

    import mlflow.sklearn

    class FakeModel:
        POSITIVE_WORDS = ("love", "helpful", "great", "good", "awesome", "best", "nice", "amazing")
        NEGATIVE_WORDS = ("garbage", "hate", "bad", "worst", "terrible", "awful", "boring", "useless")

        def predict(self, texts):
            preds = []
            for t in texts:
                low = str(t).lower()
                if any(w in low for w in self.POSITIVE_WORDS):
                    preds.append(2)  # Positive
                elif any(w in low for w in self.NEGATIVE_WORDS):
                    preds.append(0)  # Negative
                else:
                    preds.append(1)  # Neutral
            return preds

    @pytest.fixture(scope="session", autouse=True)
    def _use_fake_model():
        original = mlflow.sklearn.load_model
        mlflow.sklearn.load_model = lambda uri: FakeModel()
        yield
        mlflow.sklearn.load_model = original
