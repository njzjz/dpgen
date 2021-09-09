import os,sys,pathlib,random,json,itertools,dpdata
import numpy as np
from pathlib import Path
from typing import Set,List
from .op import OP,Status
from .opio import OPIO,DPData
from .context import IterationContext, step_model_devi, task_format, task_pattern
from .utils import link_dp_data, link_dirs
from .md_settings import MDSettings
from dpgen.generator.lib.lammps import make_lammps_input

class PrepMDLmpNative(OP):
    """This operator does the following this:
    - create path `step_model_devi`
    - create working paths for lammps molecular dynamics (MD) that estimate model devi
    - link init conf to each MD work path, and convert to lammps conf
    - link models to each MD work path
    - create lammps input file for each MD work path

    The inputs of this OP are `"init_conf"` and `"models"`
    The Paths in `"init_conf"` should be able to convert to `dpdata.System`
    The Paths in `"models"` should be valid DeePMD-kit frozen models

    The outputs of this OP are `"model_devi_dirs"`
    For example {`iter.000001/01.model_devi/md.000001`, `iter.000001/01.model_devi/md.000002`}

    Parameters
    ----------
    context : IterationContext
                DP-GEN iteration context
    init_conf: Set[Path]
                The initial configurations
    models: Set[Path]
                The models 
    mdsett : MDSettings
                The thermodynamic state of the lmp simulations
    mass_map: List[float]
                The mass of each type of atom, should match number of types in init_conf    
    append : bool
                If there is a existing model deviation work path, 
                do we append more tasks to it or
                to backup it and generate a new work path. 
    conf_format : str
                format of the configration files given in `init_conf`. 
                by default automatically detected by `dpdata`.
    shuffle_atoms : bool
                shuffle atomic coords in the configuration
    """
    @OP.set_status(status = Status.INITED)
    def __init__(
            self,
            context : IterationContext,
            mdsett : MDSettings,
            mass_map : List[float],
            append : bool = False,
            conf_format : str = "auto",
            shuffle_atoms : bool = False,
    )->None:
        super().__init__(context)
        self.append = append
        self.mdsett = mdsett
        self.shuffle_atoms = shuffle_atoms
        self.conf_format = conf_format
        self.mass_map = mass_map
        self.temps = self.mdsett.temps if self.mdsett.temps is not None else [None]
        self.press = self.mdsett.press if self.mdsett.press is not None else [None]
        self.numb_tasks_per_conf = 0
        for tt,pp in itertools.product(self.temps, self.press):
            self.numb_tasks_per_conf += 1

    @property
    def work_path(self):
        return self.context.iter_path / step_model_devi


    def get_static_input(self):
        return None
    
    def get_static_output(self):
        if self.status is not Status.EXECUTED:
            raise RuntimeError('cannot get output before the OP is executed')
        return OPIO({
            "model_devi_dirs" : set(self.all_tasks)
        })

    def _create_path(
            self,
    )->None:
        OP.create_path(self.work_path, exists_ok = self.append)

    def _task_start_idx(
            self,
    )->int:
        task_start_idx = 0
        if self.append :
            # search if there is any task in the work_path
            all_tasks = sorted([str(ii) for ii in self.work_path.glob(task_pattern)])
            # the case if any matches
            if len(all_tasks) > 0:
                task_start_idx = int(all_tasks[-1].split('.')[-1]) + 1
        return task_start_idx

    def _get_task_paths(
            self,
            task_start_idx = 0,
    )->None:
        all_tasks = []
        idx = task_start_idx
        for ii in self.init_conf:
            for jj in range(self.numb_tasks_per_conf):
                # task_path
                task_path = self.work_path / (task_format % idx) ; idx += 1
                all_tasks.append(task_path)        
        return sorted(all_tasks)

    @staticmethod
    def shuffle_conf(
            ss : dpdata.System
    )->dpdata.System:
        ss.data['coords'] = ss.data['coords'][:,np.random.permutation(ss.get_natoms()),:]
        return ss

    def _prepare_confs(
            self,
            task_start_idx = 0,
    )->None:
        idx = 0
        for ii in self.init_conf:
            for jj in range(self.numb_tasks_per_conf):
                # task_path
                task_path = self.all_tasks[idx] ; idx += 1
                if task_path.is_dir() : 
                    raise RuntimeError('f{task_path} should not exists, something wrong')
                task_path.mkdir()
                # file name of the conf
                filename = ii.name
                # protect file name if it is equal to 'conf.lmp'
                if filename == 'conf.lmp':
                    filename = Path('conf.orig.lmp')
                # create symlink to abs path of conf
                task_conf = task_path / filename
                task_conf.symlink_to(ii.resolve())
                ss = dpdata.System(task_conf, self.conf_format)
                if self.shuffle_atoms:
                    ss = PrepMDLmpNative.shuffle_conf(ss)
                if self.mdsett.no_pbc:
                    ss.remove_pbc()
                # write lmp
                task_lmp_conf = task_path / 'conf.lmp'
                ss.to('lammps/lmp', str(task_lmp_conf))

            
    def _prepare_work_model_files(
            self,
    )->None:        
        self.model_names = []
        for idx,jj in enumerate(self.models):
            self.model_names.append(Path('graph.%03d.pb' % idx))
        for idx,jj in enumerate(self.models):
            work_model = self.work_path / self.model_names[idx]
            if not work_model.exists():
                work_model.symlink_to(os.path.relpath(jj, work_model.parent))

    def _prepare_task_model_files(
            self,
    )->None:
        for idx,jj in enumerate(self.models):
            work_model = self.work_path / self.model_names[idx]
            for ii in self.all_tasks:
                task_model = ii / self.model_names[idx]
                task_model.symlink_to(os.path.relpath(work_model, task_model.parent))

    @staticmethod
    def make_input(
            tt, pp,
            mdsett,
            task_path,
            task_model_list,            
            mass_map,
            deepmd_version,
    )->str:
        file_c = make_lammps_input(
            mdsett.ens,
            'conf.lmp',
            [str(ii) for ii in task_model_list],
            mdsett.nsteps,
            mdsett.dt,
            mdsett.neidelay,
            mdsett.trj_freq,
            mass_map,
            tt,
            { 
                'use_clusters' : mdsett.use_clusters,
                'use_relative' : mdsett.relative_epsilon is not None,
                'relative' : mdsett.relative_epsilon,
                'use_relative_v' : mdsett.relative_v_epsilon is not None,
                'relative_v' : mdsett.relative_v_epsilon,
            },
            tau_t = mdsett.tau_t,
            pres = pp,
            tau_p = mdsett.tau_p,
            pka_e = mdsett.pka_e,
            ele_temp_f = mdsett.ele_temp_f,
            ele_temp_a = mdsett.ele_temp_a,
            nopbc = mdsett.no_pbc,
            deepmd_version = deepmd_version,
            trj_seperate_files = False,
        )
        return file_c

    
    def _prepare_md_settings(
            self,
    )->None:
        deepmd_version = '1.0'
        idx = 0
        for ii in self.init_conf:        
            for tt,pp in itertools.product(self.temps, self.press):
                # task_path
                task_path = self.all_tasks[idx] ; idx += 1
                file_c = PrepMDLmpNative.make_input(
                    tt, pp,
                    self.mdsett,
                    task_path,
                    self.model_names,
                    self.mass_map,
                    deepmd_version,
                )
                (task_path / 'in.lammps').write_text(file_c)
                (task_path / 'job.json').write_text(self.mdsett.to_str())


    @OP.set_status(status = Status.EXECUTED)
    def execute(
            self,
            op_in : OPIO,
    ) -> OPIO:
        self.init_conf = sorted(list(op_in['init_conf']))
        self.models = sorted(list(op_in['models']))
        task_start_idx = self._task_start_idx()
        self.all_tasks = self._get_task_paths(task_start_idx)
        self._create_path()
        self._prepare_confs()
        self._prepare_work_model_files()
        self._prepare_task_model_files()
        self._prepare_md_settings()
        return OPIO({
            "model_devi_dirs" : set(self.all_tasks)
        })
        
        
