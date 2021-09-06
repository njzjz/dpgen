import os,sys,json,glob,shutil
import dpdata
import numpy as np
import unittest
import uuid
from pathlib import Path
from mock import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
__package__ = 'ops'

from dpgen.ops.context import IterationContext
from dpgen.ops.prep_dp_train import PrepDPTrain
from dpgen.ops.opio import OPIO,DPData
from dpgen.ops.op import Status


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

class faked_rg():
    faked_random = -1
    @classmethod
    def randrange(cls,xx):
        cls.faked_random += 1
        return cls.faked_random

class TestPrepTrain(unittest.TestCase):
    def setUp(self):
        self.iter_dirs = ['iter.000000/02.fp/data.000000', 'iter.000001/02.fp/data.000001']
        self.init_dirs = ['init/foo/', 'init/bar/baz']
        all_data = [Path(ii) for ii in self.init_dirs + self.iter_dirs]
        for ii in all_data:
            ii.mkdir(parents=True)
            (ii/'type.raw').write_text(str(uuid.uuid4()))
        self.cur_iter = 2
        self.context = IterationContext('.', self.cur_iter)
        self.numb_models = 4        
        self.ptrain = PrepDPTrain(self.context,
                                  template_script,
                                  set([Path(ii) for ii in self.init_dirs]),
                                  set([Path(ii) for ii in self.iter_dirs]),
                                  self.numb_models,
        )
        self.assertEqual(self.ptrain.status, Status.INITED)

    def tearDown(self):
        dirs = ['iter.000000', 'iter.000001', 'iter.000002', 'init']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_train_script_rand_seed(self):
        faked_rg.faked_random = -1
        with mock.patch('random.randrange', faked_rg.randrange):
            self.ptrain.execute()
        for ii in range(self.numb_models):
            with open(f'iter.{self.cur_iter:06d}/00.train/train.{ii:03d}/input.json') as fp:
                jdata = json.load(fp)
            self.assertEqual(jdata['model']['descriptor']['seed'], 3*ii+0)
            self.assertEqual(jdata['model']['fitting_net']['seed'], 3*ii+1)
            self.assertEqual(jdata['training']['seed'], 3*ii+2)

    def test_input(self):
        myinput = self.ptrain.get_input()
        self.assertEqual(set(myinput.keys()), set({'init_data', 'iter_data'}))
        self.assertEqual(myinput['init_data'], set((Path(ii) for ii in self.init_dirs)))
        self.assertEqual(myinput['iter_data'], set((Path(ii) for ii in self.iter_dirs)))

    def test_output(self):
        myoutput = self.ptrain.get_output()
        self.assertEqual(set(myoutput.keys()), set({'train_dirs'}))
        ref_dirs = []
        for ii in range(self.numb_models):
            ref_dirs.append(f'iter.{self.cur_iter:06d}/00.train/train.{ii:03d}')
        ref_dirs = [Path(ii) for ii in ref_dirs]
        self.assertEqual(myoutput['train_dirs'], set(ref_dirs))

    def test_mk_train_data(self):
        self.ptrain.execute()
        self.assertEqual(self.ptrain.status, Status.EXECUTED)
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


class TestPrepTrainNoIter(unittest.TestCase):
    def setUp(self):
        self.iter_dirs = None
        self.init_dirs = ['init/foo/', 'init/bar/baz']
        all_data = [Path(ii) for ii in self.init_dirs]
        for ii in all_data:
            ii.mkdir(parents=True)
            (ii/'type.raw').write_text(str(uuid.uuid4()))
        self.cur_iter = 2
        self.context = IterationContext('.', self.cur_iter)
        self.numb_models = 4        
        self.ptrain = PrepDPTrain(self.context,
                                  template_script,
                                  set([Path(ii) for ii in self.init_dirs]),
                                  None,
                                  self.numb_models,
        )

    def tearDown(self):
        dirs = ['iter.000000', 'iter.000001', 'iter.000002', 'init']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_input(self):
        myinput = self.ptrain.get_input()
        self.assertEqual(set(myinput.keys()), set({'init_data', 'iter_data'}))
        self.assertEqual(myinput['init_data'], set((Path(ii) for ii in self.init_dirs)))
        self.assertEqual(myinput['iter_data'], None)

    def test_output(self):
        myoutput = self.ptrain.get_output()
        self.assertEqual(set(myoutput.keys()), set({'train_dirs'}))
        ref_dirs = []
        for ii in range(self.numb_models):
            ref_dirs.append(f'iter.{self.cur_iter:06d}/00.train/train.{ii:03d}')
        ref_dirs = [Path(ii) for ii in ref_dirs]
        self.assertEqual(myoutput['train_dirs'], set(ref_dirs))

    def test_mk_train_data(self):
        self.ptrain.execute()
        for ii in self.init_dirs:
            ss = ii
            for jj in range(self.numb_models):
                tt = Path('iter.000002/00.train/train.%03d/data/init_data' % jj)/ii
                self.assertEqual((Path(ss)/'type.raw').read_text(), 
                                 (Path(tt)/'type.raw').read_text())

    
class TestPrepTrainNoInit(unittest.TestCase):
    def setUp(self):
        self.iter_dirs = ['iter.000000/02.fp/data.000000', 'iter.000001/02.fp/data.000001']
        self.init_dirs = None
        all_data = [Path(ii) for ii in self.iter_dirs]
        for ii in all_data:
            ii.mkdir(parents=True)
            (ii/'type.raw').write_text(str(uuid.uuid4()))
        self.cur_iter = 2
        self.context = IterationContext('.', self.cur_iter)
        self.numb_models = 4        
        self.ptrain = PrepDPTrain(self.context,
                                  template_script,
                                  None,
                                  set([Path(ii) for ii in self.iter_dirs]),
                                  self.numb_models,
        )

    def tearDown(self):
        dirs = ['iter.000000', 'iter.000001', 'iter.000002']
        for ii in dirs:
            if Path(ii).is_dir():
                shutil.rmtree(ii)

    def test_input(self):
        myinput = self.ptrain.get_input()
        self.assertEqual(set(myinput.keys()), set({'init_data', 'iter_data'}))
        self.assertEqual(myinput['init_data'], None)
        self.assertEqual(myinput['iter_data'], set((Path(ii) for ii in self.iter_dirs)))

    def test_output(self):
        myoutput = self.ptrain.get_output()
        self.assertEqual(set(myoutput.keys()), set({'train_dirs'}))
        ref_dirs = []
        for ii in range(self.numb_models):
            ref_dirs.append(f'iter.{self.cur_iter:06d}/00.train/train.{ii:03d}')
        ref_dirs = [Path(ii) for ii in ref_dirs]
        self.assertEqual(myoutput['train_dirs'], set(ref_dirs))

    def test_mk_train_data(self):
        self.ptrain.execute()
        for ii in self.iter_dirs:
            ss = ii
            for jj in range(self.numb_models):
                tt = 'iter.000002/00.train/train.%03d/data/iter_data/' % jj + ii
                self.assertEqual((Path(ss)/'type.raw').read_text(), 
                                 (Path(tt)/'type.raw').read_text())

    
