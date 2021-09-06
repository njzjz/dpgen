import os
from abc import ABC
from pathlib import Path
from typing import Iterable, Set
from collections.abc import MutableMapping

class OPIO(MutableMapping):
    """Essentially a set of Path objects
    """    
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self._data = dict(*args, **kwargs)

    def __getitem__(
            self,
            key : str,
    ) -> Set[Path]:
        return self._data[key]

    def __setitem__(
            self,
            key : str,
            value : Iterable[Path],
    ) -> None:
        self._data[key] = set([Path(ii) for ii in value])

    def __delitem__(
            self,
            key : str,
    ) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return str(self._data)

    def keys(self):
        return self._data.keys()

    
class DPData(OPIO):
    """Build a OPIO with all valid deepmd data in path.
    Recursively search `type.raw` and treat all paths containing `type.raw`
    as valid deepmd data paths.
    """
    def __init__(
            self,
            name : str,
            path : Path,
    )->None:
        self._root_path = path
        all_type = path.rglob('type.raw')
        paths = [ii.parent for ii in all_type]
        super().__init__(name, paths)
        
    @property
    def root_path(self)->Path:
        return self._root_path 

    
