from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Model(ABC):
    @abstractmethod
    def fit(self, features: Any, target: Any, **kwargs: Any) -> Model: ...

    @abstractmethod
    def predict(self, features: Any) -> Any: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Model: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...
