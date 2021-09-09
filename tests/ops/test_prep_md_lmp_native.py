import os,sys,json,glob,shutil,textwrap,itertools
import dpdata
import numpy as np
import unittest
import uuid
from pathlib import Path
from mock import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
__package__ = 'ops'

from dpgen.ops.context import IterationContext
from dpgen.ops.prep_md_lmp_native import PrepMDLmpNative
from dpgen.ops.opio import OPIO
from dpgen.ops.op import Status
from dpgen.ops.md_settings import MDSettings

poscar_file_0=textwrap.dedent("""Al
1.0
6.6326952 0.0 0.0
0.1301009 6.5259342 0.0
0.0170968 -0.0156295 6.4869027
Al 
1
Cartesian
0 0 0 
"""
)

lmp_file_0="""
1 atoms
1 atom types
   0.0000000000    6.6326952000 xlo xhi
   0.0000000000    6.5259342000 ylo yhi
   0.0000000000    6.4869027000 zlo zhi
   0.1301009000    0.0170968000   -0.0156295000 xy xz yz

Atoms # atomic

     1      1    0.0000000000    0.0000000000    0.0000000000
"""

poscar_file_1=textwrap.dedent("""AlMg
1.0
6.6326952 0.0 0.0
0.1301009 6.5259342 0.0
0.0170968 -0.0156295 6.4869027
Al Mg
1 1
Cartesian
0 0 0 
1 1 2
"""
)

lmp_file_1="""
2 atoms
2 atom types
   0.0000000000    6.6326952000 xlo xhi
   0.0000000000    6.5259342000 ylo yhi
   0.0000000000    6.4869027000 zlo zhi
   0.1301009000    0.0170968000   -0.0156295000 xy xz yz

Atoms # atomic

     1      1    0.0000000000    0.0000000000    0.0000000000
     2      2    1.0000000000    1.0000000000    2.0000000000
"""

lmp_file_1p="""
2 atoms
2 atom types
   0.0000000000    6.6326952000 xlo xhi
   0.0000000000    6.5259342000 ylo yhi
   0.0000000000    6.4869027000 zlo zhi
   0.1301009000    0.0170968000   -0.0156295000 xy xz yz

Atoms # atomic

     1      1    1.0000000000    1.0000000000    2.0000000000
     2      2    0.0000000000    0.0000000000    0.0000000000
"""

in_lmp_template="""variable        NSTEPS          equal 1000
variable        THERMO_FREQ     equal 10
variable        DUMP_FREQ       equal 10
variable        TEMP            equal %f
variable        PRES            equal %f
variable        TAU_T           equal 0.100000
variable        TAU_P           equal 0.500000

units           metal
boundary        p p p
atom_style      atomic

neighbor        1.0 bin

box          tilt large
if "${restart} > 0" then "read_restart dpgen.restart.*" else "read_data conf.lmp"
change_box   all triclinic
mass            1 27.000000
mass            2 24.000000
pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out 
pair_coeff      

thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz
thermo          ${THERMO_FREQ}
dump            1 all custom ${DUMP_FREQ} dump.traj id type x y z fx fy fz
restart         10000 dpgen.restart

if "${restart} == 0" then "velocity        all create ${TEMP} %d"
fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}

timestep        0.002000
run             ${NSTEPS} upto
"""

class faked_rg():
    @classmethod
    def randrange(cls,xx):        
        return 1

class faked_rg_perm():
    outputs = [ [0, 1], [0, 1], [1, 0], [1, 0], 
                [0, 1], [1, 0], [1, 0], [0, 1],
                [1, 0], [1, 0], [0, 1], [0, 1], 
    ]
    idx = 0
    @classmethod
    def permutation(cls,xx):        
        if xx == 1:
            return [0]
        elif xx == 2:
            ret = cls.outputs[cls.idx]; cls.idx += 1
            return ret
        else :
            raise RuntimeError('invalid xx for this mock')


class TestPrepMDLmpNative(unittest.TestCase):
    def setUp(self):
        self.init_conf = ['init/0/POSCAR', 'init/1/POSCAR']
        self.poscars = [poscar_file_0, poscar_file_1]
        self.models = [
            'iter.000001/00.train/train.000/frozen_model.pb', 
            'iter.000001/00.train/train.001/frozen_model.pb',
        ]
        self.cur_iter = 1
        self.init_conf = [Path(ii) for ii in self.init_conf]
        self.models = [Path(ii) for ii in self.models]
        for ii,jj in zip(self.init_conf, self.poscars):
            ii.parent.mkdir(parents=True)
            ii.write_text(jj)
        for ii in self.models:
            ii.parent.mkdir(parents=True)
            ii.write_text(str(uuid.uuid4()))
        self.mdsett = MDSettings(
            'npt',
            0.002,
            1000,
            10,
            temps = [100, 200, 300],
            press = [10, 1000],
        )

        self.context = IterationContext('.', self.cur_iter)
        self.pmd = PrepMDLmpNative(
            self.context,
            self.mdsett,
            [27., 24.],
            append = False,
            conf_format = 'vasp/poscar',
            shuffle_atoms = False,
        )
        self.pmd_1 = PrepMDLmpNative(
            self.context,
            self.mdsett,
            [27., 24.],
            append = True,
            conf_format = 'vasp/poscar',
            shuffle_atoms = False,
        )
        self.pmd_2 = PrepMDLmpNative(
            self.context,
            self.mdsett,
            [27., 24.],
            append = False,
            conf_format = 'vasp/poscar',
            shuffle_atoms = True,
        )
        self.input = OPIO({
            'init_conf' : set(self.init_conf),
            'models' : set(self.models),
        })

        
    def tearDown(self):
        dirs = ['iter.000001', 'init']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_input(self):
        myinput = self.input
        self.assertEqual(set(myinput.keys()), set({'init_conf', 'models'}))
        self.assertEqual(myinput['init_conf'], set(self.init_conf))
        self.assertEqual(myinput['models'], set(self.models))

    def test_output(self):
        myoutput = self.pmd.execute(self.input)
        self.assertEqual(set(myoutput.keys()), set({'model_devi_dirs'}))
        ref_dirs = []
        for ii in range(len(self.init_conf) * len(self.mdsett.temps) * len(self.mdsett.press)):
            ref_dirs.append(f'iter.{self.cur_iter:06d}/01.model_devi/task.{ii:06d}')
        ref_dirs = [Path(ii) for ii in ref_dirs]
        self.assertEqual(myoutput['model_devi_dirs'], set(ref_dirs))

    def test_exec(self):
        with mock.patch('random.randrange', faked_rg.randrange):
            myoutput = self.pmd.execute(self.input)
        all_tasks = myoutput['model_devi_dirs']
        self.assertEqual(self.pmd.status, Status.EXECUTED)
        # check content of conf file
        all_tasks = sorted([str(ii) for ii in all_tasks])
        for ii in range(len(all_tasks) // 2):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, lmp_file_0)
        for ii in range(len(all_tasks) // 2, len(all_tasks)):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, lmp_file_1)
        # check model files
        for ii in all_tasks:
            task_models = sorted(Path(ii).glob('graph*pb'))
            for jj,kk in zip(task_models, self.models):
                self.assertEqual(jj.read_text(), kk.read_text())
        # check lammps in
        model_str = ''
        for idx,ii in enumerate(self.models):
            model_str += f'graph.{idx:03d}.pb '
        for ii,(tt,pp) in zip(all_tasks, itertools.product(self.mdsett.temps, self.mdsett.press)):
            in_lmp = (Path(ii) / 'in.lammps').read_text()
            self.assertEqual(in_lmp, in_lmp_template % (tt, pp, model_str, faked_rg.randrange(0)+1))


    def test_exec_append(self):
        with mock.patch('random.randrange', faked_rg.randrange):
            self.pmd.execute(self.input)
            myoutput = self.pmd_1.execute(self.input)
        all_tasks = myoutput['model_devi_dirs']
        all_tasks = sorted([str(ii) for ii in all_tasks])
        ntasks = len(self.init_conf) * len(self.mdsett.temps) * len(self.mdsett.press)            
        ref_all_tasks = [f'iter.{self.cur_iter:06d}/01.model_devi/task.{ii+ntasks:06d}'
                         for ii in range(ntasks)]
        self.assertEqual(all_tasks, ref_all_tasks)
        # check content of conf file
        for ii in range(len(all_tasks) // 2):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, lmp_file_0)
        for ii in range(len(all_tasks) // 2, len(all_tasks)):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, lmp_file_1)        
        # check model files
        for ii in all_tasks:
            task_models = sorted(Path(ii).glob('graph*pb'))
            for jj,kk in zip(task_models, self.models):
                self.assertEqual(jj.read_text(), kk.read_text())
        # check lammps in
        model_str = ''
        for idx,ii in enumerate(self.models):
            model_str += f'graph.{idx:03d}.pb '
        for ii,(tt,pp) in zip(all_tasks, itertools.product(self.mdsett.temps, self.mdsett.press)):
            in_lmp = (Path(ii) / 'in.lammps').read_text()
            self.assertEqual(in_lmp, in_lmp_template % (tt, pp, model_str, faked_rg.randrange(0)+1))


    def test_exec_shuffle(self):
        with mock.patch('random.randrange', faked_rg.randrange):
            with mock.patch('numpy.random.permutation', faked_rg_perm.permutation):
                myoutput = self.pmd_2.execute(self.input)
        all_tasks = myoutput['model_devi_dirs']
        all_tasks = sorted([str(ii) for ii in all_tasks])
        ntasks = len(self.init_conf) * len(self.mdsett.temps) * len(self.mdsett.press)            
        ref_all_tasks = [f'iter.{self.cur_iter:06d}/01.model_devi/task.{ii:06d}'
                         for ii in range(ntasks)]
        self.assertEqual(all_tasks, ref_all_tasks)
        # check content of conf file
        for ii in range(len(all_tasks) // 2):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, lmp_file_0)
        ref_confs = [ lmp_file_1, lmp_file_1, lmp_file_1p, lmp_file_1p,
                      lmp_file_1, lmp_file_1p, lmp_file_1p, lmp_file_1,
                      lmp_file_1p, lmp_file_1p, lmp_file_1, lmp_file_1,
        ]
        for ii,jj in zip(range(len(all_tasks) // 2, len(all_tasks)), ref_confs):
            fc = (Path(all_tasks[ii])/'conf.lmp').read_text()
            self.assertEqual(fc, jj)
        
    def test_make_input(self):
        print('test_make_input to be implemented!!!!')
