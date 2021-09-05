"""Provide Paths classes."""
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Iterable

class OPPath(ABC):
    """OP path.
    
    Parameters
    ----------
    path : str
        the path of file, can be static or dynamic
    dynamic : bool, optional
        dynamic path or not. If not set, it will be decided by whether
        the path contains glob patterns
    
    Examples
    --------
    >>> type(OPPath("path.py")).__name__
    StaticPath
    >>> type(OPPath("*.py")).__name__
    DynamicPath
    >>> type(OPPath("path.py", dynamic=True)).__name__
    DynamicPath
    >>> type(OPPath("*.py", dynamic=False)).__name__
    StaticPath
    """
    def __init__(self, path: str, dynamic: bool = None) -> None:
        self.path = str(path)
    
    def __new__(cls, path: str, *args, dynamic: bool = None, **kwargs):
        if dynamic is not None:
            new_cls = DynamicPath if dynamic else StaticPath
        elif "*" in path:
            new_cls = DynamicPath
        else:
            new_cls = StaticPath
        return object.__new__(new_cls)

    def __str__(self) -> str:
        return self.path
    
    def __hash__(self):
        return hash(self.path)
    
    def __eq__(self, other):
        """Compare if two paths are equal.
        
        Examples
        --------
        >>> OPPath("path.py") == OPPath("path.py")
        True
        >>> OPPath("path.py") == OPPath("path.py", dynamic=True)
        False
        """
        if isinstance(other, type(self)):
            return self.path == other.path
        return False
    
    @abstractmethod
    def get_all_files(self, dir: Path = None) -> List[Path]:
        """Returns all files under the base directory.
        
        Parameters
        ----------
        dir : pathlib.Path, default: current working directory
            base directory
        """


class StaticPath(OPPath):
    """Static path stores a fixed path.
    
    Parameters
    ----------
    path : str
        the path of file, can be static or dynamic
    
    Examples
    --------
    >>> DynamicPath("path.py")
    """
    def get_all_files(self, dir: Path = None) -> List[Path]:
        """Returns all files under the base directory.
        
        Parameters
        ----------
        dir : pathlib.Path, default: current working directory
            base directory
        
        Examples
        --------
        >>> DynamicPath("*.py").get_all_files()
        """
        if dir is None:
            dir = Path.cwd()
        return [dir/self.path]


class DynamicPath(OPPath):
    """Dynamic path stores the dynamic number of paths, using glob
    to find files.

    Parameters
    ----------
    path : str
        the path of file, can be static or dynamic
    
    Examples
    --------
    >>> DynamicPath("*.py")
    """
    def get_all_files(self, dir: Path = None) -> List[Path]:
        """Returns all files under the base directory.
        
        Parameters
        ----------
        dir : pathlib.Path, default: current working directory
            base directory
        
        Examples
        --------
        >>> DynamicPath("*.py").get_all_files()
        """
        if dir is None:
            dir = Path.cwd()
        return sorted(dir.glob(self.path))


class OPPathSet(frozenset):
    """The set of OP Paths.
    
    Examples
    --------
    >>> OPPathSet((OPPath("path.py"), OPPath("*.py")))
    >>> OPPathSet(("path.py", "*.py"))
    """
    def __new__(cls, iterable: Iterable[OPPath]):
        iterable = [x if isinstance(x, OPPath) else OPPath(x) for x in iterable]
        return super().__new__(cls, iterable)

    def get_all_files(self, dir: Path = None) -> List[Path]:
        """Returns all files in the set under the base directory.
        
        Parameters
        ----------
        dir : pathlib.Path, default: current working directory
            base directory
        
        Examples
        --------
        >>> OPPathSet((OPPath("haha"), OPPath("*.py"))).get_all_files()
        """
        if dir is None:
            dir = Path.cwd()
        return sorted(list(set(sum([pp.get_all_files() for pp in self], []))))
