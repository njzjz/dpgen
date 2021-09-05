import os,sys,json,glob,shutil
import dpdata
import numpy as np
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
__package__ = 'ops'

from dpgen.ops.context import IterationContext
from dpgen.ops.prep_train import PrepDPTrain
from dpgen.ops.opio import OPIO,DPData


template_script = \
{
    "model" : {
	"descriptor": {
            "type":     "se_e2_a",
	    "seed":	1
	},
	"fitting_net" : {
	    "seed":	1
	}
    },
    "training" : {
	"systems":	[],
	"set_prefix":	"set",
	"stop_batch":	2000,
	"batch_size":	'auto',
	"seed":		1,
    },
}

class TestLinkDataAbs(unittest.TestCase):
    def setUp(self):
        self.iter_dirs = ['iter.000000/02.fp/data.000000', 'iter.000001/02.fp/data.000001']
        self.init_dirs = ['init/foo/', 'init/bar/baz']
        all_data = [Path(ii) for ii in self.init_dirs + self.iter_dirs]
        for ii in all_data:
            ii.mkdir(parents=True)
            (ii/'type.raw').write_text(str(uuid.uuid4()))
        self.context = IterationContext('.', 2)
        self.numb_models = 4        
        self.ptrain = PrepDPTrain(self.context,
                                  template_script,
                                  OPIO('init_data', self.init_dirs),
                                  OPIO('iter_data', self.iter_dirs),
                                  self.numb_models,
        )

    def tearDown(self):
        dirs = ['iter.000000', 'iter.000001', 'iter.000002', 'init']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_mk_train_script(self):
        pass

    def test_mk_train_data(self):
        self.ptrain.execute()
        for ii in self.init_dirs:
            ss = ii
            for jj in range(self.numb_models):
                tt = Path('iter.000002/00.train/train.%03d/data/init_data' % jj)/ii
                self.assertEqual((Path(ss)/'type.raw').read_text(), 
                                 (Path(tt)/'type.raw').read_text())
        for ii in self.iter_dirs:
            ss = ii
            for jj in range(self.numb_models):
                tt = 'iter.000002/00.train/train.%03d/data/iter_data/' % jj + ii
                self.assertEqual((Path(ss)/'type.raw').read_text(), 
                                 (Path(tt)/'type.raw').read_text())

