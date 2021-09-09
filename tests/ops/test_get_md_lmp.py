import os,sys,json,glob,shutil,textwrap,itertools
import dpdata
import numpy as np
import unittest
import uuid
from pathlib import Path
from mock import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
__package__ = 'ops'

from dpgen.ops.context import Context
from dpgen.ops.get_md_lmp import GetMaxMDLmp
from dpgen.ops.opio import OPIO
from dpgen.ops.op import Status

class TestGetMaxMDLmp(unittest.TestCase):
    def setUp(self):
        self.work_path = Path('iter.000001/01.model_devi/task.000000')
        self.md_path = self.work_path / 'model_devi.out'
        self.fi = """# comment
           0       3.150048e-01       3.581620e-02       2.611629e-01       8.413737e-02       2.195287e-02       6.787650e-02
          10       3.145855e-01       3.373956e-02       2.652665e-01       1.684026e-01       7.627893e-02       1.051007e-01
          20       3.329576e-01       3.275904e-02       2.735763e-01       2.224489e-01       8.436380e-02       1.198283e-01
          30       3.945782e-01       3.785559e-02       2.939188e-01       2.445201e-01       6.921637e-02       1.541938e-01
          40       5.085895e-01       3.972635e-02       3.446580e-01       2.166893e-01       8.677478e-02       1.343453e-01
        """
        self.ref_fo = [
                8.413737e-02, 3.150048e-01,      
                1.684026e-01, 3.145855e-01,      
                2.224489e-01, 3.329576e-01,      
                2.445201e-01, 3.945782e-01,      
                2.166893e-01, 5.085895e-01,      
        ]
        self.work_path.mkdir(parents=True)
        self.md_path.write_text(self.fi)
        self.context = Context('.')
        self.gmd = GetMaxMDLmp(self.context, self.work_path)

    def tearDown(self):
        dirs = ['iter.000001']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)
        
    def test_static_input(self):
        myinput = self.gmd.get_static_input()
        self.assertEqual(myinput._data, {'md_path' : self.work_path/'model_devi.out'})

    def test_static_output(self):
        myinput = self.gmd.get_static_output()
        self.assertEqual(myinput._data, {'max_md_path' : self.work_path/'max_model_devi.out'})

    def test_exec(self):
        self.gmd.execute(None)
        out = np.loadtxt(self.gmd.get_static_output()['max_md_path'])
        ref_out = np.array(self.ref_fo)
        np.testing.assert_almost_equal(out.ravel(), ref_out.ravel())
