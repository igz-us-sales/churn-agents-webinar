import warnings
from typing import List

import mlrun
import numpy as np
from cloudpickle import load
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")


class ChurnModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> List:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict_proba(feats)
        # Only interested in churn likelihood
        return [i[1] for i in result.tolist()]
