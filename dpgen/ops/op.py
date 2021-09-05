import abc,os,functools
from typing import List,Set
from abc import ABC
from enum import Enum
from pathlib import Path
from .opio import OPIO
from .context import Context

class Status(object):
    """The status of DP-GEN events
    """
    INITED = 0
    EXECUTED = 1
    ERROR = 2

class OP(ABC):
    """The OP of DP-GEN. The OP is defined as an operation that has some
    effects on the files in the system. One can get all the files
    needed for the OP by `get_input` and all files output by
    `get_output`. The action of the OP is activated by `execute`.

    """
    def __init__(
            self,
            context,
    )->None:
        self._context = context

    @property
    def context(self):
        return self._context
    
    @property
    def status(self):
        return self._status
    
    @property
    def work_path (self):
        return self._work_path

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

    def set_status(status):
        def decorator_set_status(func):
            @functools.wraps(func)
            def wrapper_set_status(self, *args, **kwargs):
                func(self, *args, **kwargs)
                self._status = status
            return wrapper_set_status
        return decorator_set_status


class StaticOP(OP):
    """A DP-GEN OP. Know its input and output after initialization
    """
    def __init__(
            self,
            context : Context,
            work_path : Path,
            op_input : Set[Path],
            op_output : Set[Path],
    )->None:
        super().__init__(context)
        self._work_path = work_path
        self._input = op_input
        self._output = op_output
        
    def get_input(self) -> Set[Path]:
        return self._input
    
    def get_output(self) -> Set[Path]:
        return self._output


class DynamicOP(OP):
    """A DP-GEN OP. Only know its input and output after the OP is executed.
    """
    def __init__(
            self,
            context,
            work_path,
    )->None:
        super().__init__(context)
        self._work_path = work_path
        self.opctrl = OPController(self)
        
    def get_input(self) -> Set[Path]:
        if self.status is not Status.EXECUTED:
            raise RuntimeError('Dynamic OP can only get input after it is executed.')
        return self.opctrl.get_input()

    def get_output(self) -> Set[Path]:
        if self.status is not Status.EXECUTED:
            raise RuntimeError('Dynamic OP can only get output after it is executed.')
        return self.opctrl.get_output()


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
