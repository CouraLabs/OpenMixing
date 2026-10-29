from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..core.decode import decode


class AudioCache:
    def __init__(self, size: int = 6):
        self.size, self._d = size, OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._d:
            self._d.move_to_end(path)
            return self._d[path]
        a = decode(path)
        self._d[path] = a
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return a
