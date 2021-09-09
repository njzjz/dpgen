import os,sys,pathlib,random,json,itertools,dpdata
import numpy as np
from pathlib import Path
from typing import Set,List
from .op import OP,Status
from .opio import OPIO,DPData
from .context import Context

class GetMaxMDLmp(OP):
    """This operator extract maximum model deviation from lammps output
    `model_devi.out` and dump it to max_model_devi.out
    """
    @OP.set_status(status = Status.INITED)
    def __init__(
            self,
            context : Context,
            work_path : Path,
    ):
        super().__init__(context)
        self._work_path = work_path
        self.md_path = self.work_path / 'model_devi.out'
        self.max_md_path = self.work_path / 'max_model_devi.out'

    @property
    def work_path(self):
        return self._work_path

    def get_static_input(self):
        return OPIO({ 
            "md_path" : self.md_path,
        })
    
    def get_static_output(self):
        return OPIO({ 
            "max_md_path" : self.max_md_path
        })
    
    @OP.set_status(status = Status.EXECUTED)
    def execute(
            self,
            op_in : OPIO
    ) -> OPIO:
        dd = np.loadtxt(self.md_path)
        dd = dd[:,(4,1)]
        np.savetxt(self.max_md_path, dd)
        return None
