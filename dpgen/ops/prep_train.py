import os,sys,pathlib,random,json
from pathlib import Path
from .op import OP
from .opio import OPIO,DPData
from .context import IterationContext, iteration_pattern, iterdata_pattern, train_pattern, step_fp, step_train, train_format
from .utils import create_path, link_dp_data, link_dirs

class PrepDPTrain(OP): 
    r"""This operator will do the following things:
    - create path step_train
    - link the init training data to 00.train/init_data using abs path
    - link the data generated by dpgen to 00.train/iter_data using rel path
    - make the input trianing script. 
    
    Parameters
    ----------
    template_script
                template of the training input script
    init_data
                path to the initial training data
    context
                DP-GEN iteration context
    """
    def __init__(
            self,
            context : IterationContext,
            template_script : dict,
            init_data : OPIO = None,
            iter_data : OPIO = None,
            numb_models : int = 4,
    )->None:
        super().__init__(context)
        self.training_data_pattern = os.path.join(iteration_pattern, step_fp, iterdata_pattern)        
        self.init_data = init_data
        self.iter_data = iter_data
        self.script = template_script
        self.numb_models = numb_models

    def _create_path(
            self,
    )->None:
        create_path(self.work_path)

    def _link_init_data(
            self,
    )->None:
        if self.init_data is None:
            return
        init_data_dir = self.work_path / 'init_data'
        if init_data_dir.exists():
            raise RuntimeError('init_data dir should not exists, something wrong')
        link_dirs(
            self.init_data.value,
            init_data_dir, 
            link_abspath = True,
        )

    def _link_iter_data(
            self,
    )->None:
        if self.iter_data is None:
            return
        iter_data_dir = self.work_path / 'iter_data'
        if iter_data_dir.exists():
            raise RuntimeError('iter_data dir should not exists, something wrong')
        link_dirs(
            self.iter_data.value, 
            iter_data_dir, 
            link_abspath = False, 
            data_path_pattern = os.path.join(iteration_pattern, 
                                             step_fp,
                                             iterdata_pattern),
        )

    def _make_train_dirs(self):
        for ii in range(self.numb_models):
            train_dir = self.work_path / (train_format % ii)
            data_dir = train_dir / 'data'
            train_dir.mkdir()
            data_dir.mkdir()
            (data_dir / 'init_data').symlink_to(Path('..')/'..'/'init_data')
            (data_dir / 'iter_data').symlink_to(Path('..')/'..'/'iter_data')

    def _make_train_script(self):
        for ii in range(self.numb_models):
            fname = self.work_path / (train_format % ii) / 'input.json'
            self._make_train_script_ii(ii, fname)

    def _make_train_script_ii(self, ii, fname):
        jtmp = self.script
        jtmp['systems'] = 'data'
        if jtmp['model']['descriptor']['type'] == 'hybrid':
            for desc in jtmp['model']['descriptor']['list']:
                desc['seed'] = random.randrange(sys.maxsize) % (2**32)
        else:
            jtmp['model']['descriptor']['seed'] = random.randrange(sys.maxsize) % (2**32)
        jtmp['model']['fitting_net']['seed'] = random.randrange(sys.maxsize) % (2**32)
        jtmp['training']['seed'] = random.randrange(sys.maxsize) % (2**32)
        with open(fname, 'w') as fp:
            json.dump(jtmp, fp, indent = 4)

    def get_input(self):
        my_input = OPIO(self.context)
        ## add input paths
        return my_input

    def get_output(self):
        my_output = OPIO(self.context)
        ## add output paths
        my_output.add('work_path', self.work_path)
        return my_output

    @property
    def work_path(self):
        return self.context.iter_path / step_train

    def execute(self):
        # create path
        self._create_path()
        # link init data
        self._link_init_data()
        # link iter data
        self._link_iter_data()
        # mkdir 
        self._make_train_dirs()
        # make scripts 
        self._make_train_script()


# if __name__ == '__main__':

