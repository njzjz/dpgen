import abc,os
from typing import List
from abc import ABC
from enum import Enum
from pathlib import Path

class Status(object):
    """The status of DP-GEN events
    """
    INITED = 0
    EXECUTED = 1
    ERROR = 2
    DIED = 3

class OPIO(object):    
    def __init__(
            self,
            context,
    )->None:
        self._context = context


    def add(
            self,
            key : str,
            path : List[Path],
    )->None:
        """add a (key,path) pair. 
        
        Parameters
        ----------
        key
                the key
        path
                the paths to the location
        """
        self._data[key] = path
        return self


    def valid_all_path(
            self,
    ):
        for ii in self._data.keys():
            self.valid_path(ii)

    def valid_path(
            self,
            key,
    ):
        for jj in self._data[key]:
            if not jj.exists():
                raise FileNotFoundError(f"{jj} does not exists")


class OP(ABC):
    """The OP of DP-GEN. The OP is defined as an operation that has some
    effects on the files in the system. One can get all the files
    needed for the OP by `get_input` and all files output by
    `get_output`. The action of the OP is activated by `execute`.

    """
    def __init__(
            self,
            context,
    ):
        self._context = context

    @property
    def context(self):
        return self._context
    
    @property
    def status(self):
        return self._status
    
    @property
    def work_path (self):
        """The work path
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_input(self) -> OPIO:
        """Get a list of input files
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_output(self) -> OPIO:
        """Get a list of ouput files
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def execute (self) -> None:
        """Run the OP
        """
        raise NotImplementedError


class OPSet(ABC):
    """A set of ops that can be executed in parallel. The work_path of
    each op should be a subdirectory of the
    OPSet.get_work_path(). The OPs may share common input/output files.

    """

    @abc.abstractmethod
    def get_work_path (self) -> None:
        """Run the OP
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_common_input(self) -> OPIO:
        """Get a list of common input files
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_common_output(self) -> OPIO:
        """Get a list of common ouput files
        """
        raise NotImplementedError
