from zenml.materializers.base_materializer import BaseMaterializer
from typing import Any, Type
import joblib
import os

class JoblibListMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (list,)
    ASSOCIATED_ARTIFACT_TYPES = ("data",)

    def load(self, data_type: Type[Any]) -> Any:
        return joblib.load(os.path.join(self.uri, "data.pkl"))

    def save(self, data: Any) -> None:
        joblib.dump(data, os.path.join(self.uri, "data.pkl"))
