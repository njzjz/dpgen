import os,sys,json,glob,shutil
import dpdata
import numpy as np
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
__package__ = 'ops'

from dpgen.ops.utils import os_path_split, link_dp_data


class TestPathSplit(unittest.TestCase):
    def test_split_abs(self):
        aa = '/a/b/c'
        bb = os_path_split(aa)
        self.assertEqual(bb, ['/', 'a', 'b', 'c'])

    def test_split_rel(self):
        aa = 'a/b/c'
        bb = os_path_split(aa)
        self.assertEqual(bb, ['a', 'b', 'c'])

    def test_split_rel_dir(self):
        aa = 'a/b/c/'
        bb = os_path_split(aa)
        self.assertEqual(bb, ['a', 'b', 'c', ''])

class TestLinkDataAbs(unittest.TestCase):
    def setUp(self):
        self.dirs = ['source/data0/subdata0', 'source/data0/subdata1', 'source/data1', 'source/foo/bar']
        all_data = [Path(ii) for ii in self.dirs]
        for ii in all_data:
            ii.mkdir(parents=True)
            (ii/'type.raw').write_text(str(uuid.uuid4()))

    def tearDown(self):
        shutil.rmtree('source')
        shutil.rmtree('target')

    def test_link_rel(self):
        link_dp_data('source', 'target', link_abspath = False)
        for ii in self.dirs:
            ss = ii
            tt = ii.replace('source', 'target')
            self.assertEqual((Path(ss)/'type.raw').read_text(), 
                             (Path(tt)/'type.raw').read_text())

    def test_link_abs(self):
        link_dp_data('source', 'target', link_abspath = True)
        for ii in self.dirs:
            ss = ii
            tt = ii.replace('source', 'target')
            self.assertEqual((Path(ss)/'type.raw').read_text(), 
                             (Path(tt)/'type.raw').read_text())

    def test_link_abs_1(self):
        link_dp_data('source', 'target/target1', link_abspath = False)
        for ii in self.dirs:
            ss = ii
            tt = ii.replace('source', 'target/target1')
            self.assertEqual((Path(ss)/'type.raw').read_text(), 
                             (Path(tt)/'type.raw').read_text())

    def test_link_abs_pattern(self):
        link_dp_data('source', 
                     'target/target1', 
                     link_abspath = False,
                     data_path_pattern = '.*data.*'
        )
        for ii in Path('target').iterdir():
            self.assertTrue('foo' not in str(ii))
        for ii in self.dirs[:-1]:
            ss = ii
            tt = ii.replace('source', 'target/target1')
            self.assertEqual((Path(ss)/'type.raw').read_text(), 
                             (Path(tt)/'type.raw').read_text())

