from abc import abstractmethod
from pathlib import Path
from typing import List
from itertools import chain

from dpdispatcher import Machine, Resources, Task, Submission

from .op import OP, Status
from .opio import OPIO
from .context import IterationContext, task_pattern


class DispatcherOP(OP):
    r"""This OP will call DPDispatcher to execuate commands on remote.
    
    Parameters
    ----------
    context : IterationContext
        DP-GEN iteration context
    machine : dict
        DPDispatcher machine parameters. See also DPDispatcher doc.
    resources : dict
        DPDispatcher machine parameters. See also DPDispatcher doc.
    command : str
        path or command to a certain program on remote
    """
    @OP.set_status(status = Status.INITED)
    def __init__(
            self,
            context: IterationContext,
            machine: dict,
            resources: dict,
            command: str,
    )->None:
        super().__init__(context)
        self.machine = Machine(**machine)
        self.resources = Resources(**resources)
        self.user_command = command
    
    @property
    def command(self) -> str:
        """str: The full bash command produced by the command the user provides."""
        return self.user_command

    @property
    def task_pattern(self) -> str:
        """str: task pateern for glob, e.g. `task.*`"""
        return task_pattern
    
    @property
    def outlog(self) -> str:
        return "log"

    @property
    def errlog(self) -> str:
        return "log"
    
    @property
    @abstractmethod
    def forward_files(self) -> List[str]:
        """List[str]: relative path to task path"""

    @property
    @abstractmethod
    def backward_files(self) -> List[str]:
        """List[str]: relative path to task path"""
    
    @property
    def forward_common_files(self) -> List[str]:
        """List[str]: relative path to work path"""
        return []

    @property
    def backward_common_files(self) -> List[str]:
        return []
    
    @property
    def task_paths(self) -> List[Path]:
        if self.status is not Status.EXECUTED:
            raise RuntimeError
        return self.work_path.glob(self.task_pattern)
    
    def get_input(self) -> OPIO:
        return OPIO({
            "task_forward_files": set(chain.from_iterable([[pp/ff for ff in self.forward_files] for pp in self.task_paths])),
            "forward_common_files": set([self.work_path/ff for ff in self.forward_common_files])
        })

    def get_ouput(self) -> OPIO:
        return OPIO({
            "task_backward_files": set(chain.from_iterable([[pp/ff for ff in self.backward_files] for pp in self.task_paths])),
            "backward_common_files": set([self.work_path/ff for ff in self.backward_common_files])
        })

    @OP.set_status(status = Status.EXECUTED)
    def execute(self):
        tasks = []

        for task_path in self.task_paths:
            task = Task(
                command=self.command, 
                task_work_path=task_path,
                forward_files=self.forward_files,
                backward_files=self.backward_files,
                outlog=self.outlog,
                errlog=self.errlog
            )
            tasks.append(task)

        submission = Submission(
            work_base=self.work_path,
            machine=self.machine,
            resources=self.resources,
            task_list=tasks,
            forward_common_files=self.forward_common_files,
            backward_common_files=[]
        )
        submission.run_submission()
