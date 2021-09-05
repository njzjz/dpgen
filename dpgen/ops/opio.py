import os
from abc import ABC
from pathlib import Path
from typing import Iterable, Set

class OPIO(object):
    """Essentially a set of Path objects
    """    
    def __init__(
            self, 
            name : str, 
            paths : Iterable[Path],
    ):
        self._name = name
        self._paths = set({})
        for ii in paths:
            self._paths.add(Path(ii))
        
    @property
    def key(self)->str:
        return self._name

    @property
    def value(self)->Set[Path]:
        return self._paths

    
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

    
