"""Will eventually contain helpers to get metadata about models, e.g. pricing."""

from typing import NamedTuple

from .utils import clean_column_name


class ModelId(NamedTuple):
    provider: str | None
    name: str

    @classmethod
    def parse(cls, name: str) -> "ModelId":
        if "/" not in name:
            return ModelId(None, name.lower())

        provider, model = name.split("/", 1)
        return ModelId(provider.lower(), model.lower())

    def __str__(self) -> str:
        if self.provider:
            return f"{self.provider}/{self.name}"
        return self.name

    def __repr__(self) -> str:
        return f"ModelId(provider={self.provider!r}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, ModelId):
            return self.provider == other.provider and self.name == other.name
        return NotImplemented

    def column_name(self, include_provider: bool = False) -> str:
        """Return a string suitable for use as a DataFrame column name."""
        if include_provider and self.provider:
            return clean_column_name(f"{self.provider}_{self.name}")
        return clean_column_name(self.name)
