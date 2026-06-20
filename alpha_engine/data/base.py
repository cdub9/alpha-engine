"""Base interface for data providers.

Providers fetch raw data from an external source and yield typed domain
objects (MarketBar, MacroObservation, NewsEvent, etc.). Storage of the
results is handled separately by storage.py so providers stay pure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


class DataProvider(ABC, Generic[T]):
    """Base class for any data provider. Subclasses implement `fetch`."""

    name: str

    @abstractmethod
    def fetch(self, *args, **kwargs) -> Iterable[T]:
        """Yield typed records from the external source."""
        ...
