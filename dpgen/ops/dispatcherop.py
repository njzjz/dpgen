from abc import abstractmethod
from pathlib import Path
from typing import List, Set

from dpdispatcher import Machine, Resources, Task, Submission

from .op import OP, Status
from .opio import OPIO
from .context import Context


class DispatcherOP(OP):
    r"""This OP will call DPDispatcher to execuate commands on remote.
    
    Parameters
    ----------
    context : IterationContext
        DP-GEN iteration context
    machine : Machine
        DPDispatcher machine parameters. See also DPDispatcher doc.
    resources : Resources
        DPDispatcher machine parameters. See also DPDispatcher doc.
    command : str
        path or command to a certain program on remote
    """
    @OP.set_status(status = Status.INITED)
    def __init__(
            self,
            context: Context,
            machine: Machine,
            resources: Resources,
            command: str,
    )->None:
        super().__init__(context)
        self.machine = machine
        self.resources = resources
        self.user_command = command
    
    @property
    def command(self) -> str:
        """str: The full bash command produced by the command the user provides."""
        return self.user_command
    
    @property
    def outlog(self) -> str:
        return "log"

    @property
    def errlog(self) -> str:
        return "log"
    
    @property
    @abstractmethod
    def forward_files(self) -> Set[Path]:
        """List[str]: relative path to task path"""

    @property
    @abstractmethod
    def backward_files(self) -> Set[Path]:
        """List[str]: relative path to task path"""
    
    @property
    @classmethod
    def forward_common_files(cls) -> Set[Path]:
        """List[str]: relative path to work path"""
        return set()

    @property
    @classmethod
    def backward_common_files(cls) -> Set[Path]:
        return set()
    
    @classmethod
    def get_static_input(cls) -> OPIO:
        return OPIO({
            "forward_common_files": cls.forward_common_files
        })

    @classmethod
    def get_static_ouput(cls) -> OPIO:
        return OPIO({
            "backward_common_files": cls.backward_common_files
        })

    @OP.set_status(status = Status.EXECUTED)
    def execute(
            self,
            op_in : OPIO
    ) -> OPIO:
        task_paths = op_in["tasks"]
        tasks = []

        for task_path in task_paths:
            task = Task(
                command=self.command, 
                task_work_path=str(task_path),
                forward_files=self.path_to_str(self.forward_files),
                backward_files=self.path_to_str(self.backward_files),
                outlog=self.outlog,
                errlog=self.errlog,
            )
            tasks.append(task)

        submission = Submission(
            work_base=self.work_path,
            machine=self.machine,
            resources=self.resources,
            task_list=tasks,
            forward_common_files=self.path_to_str(self.forward_common_files),
            backward_common_files=self.path_to_str(self.backward_common_files),
        )
        submission.run_submission()
        return op_in

    @classmethod
    def path_to_str(cls, paths: List[Path]) -> List[str]:
        return list([str(path) for path in paths])
