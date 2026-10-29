from functools import cached_property

from .annotations import Annotations
from .catalog import Catalog
from .samples import SampleStore
from .stemstore import StemStore
from .store import Store
from .tracks import ANALYSIS_VERSION, TrackStore


class Library(Catalog, TrackStore, StemStore, Annotations, SampleStore):
    @cached_property
    def store(self) -> Store:
        return Store(self.state)
