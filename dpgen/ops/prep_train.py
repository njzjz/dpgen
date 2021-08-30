from .op import OP, OPIO

class PrepDPTrain(OP): 
    def __init__(
            self,
            template_script : dict,
            init_data_dir : str,
            context : Context,
    )->None:
        super().__init__(context)
        self.init_data_dir = [init_data_dir]
        self.iter_data_dir = self.context.all_prev_iter
        pass

    def get_input(self):
        my_input = OPIO(self.context)
        ## add input paths
        return my_input

    def get_output(self):
        my_output = OPIO(self.context)
        ## add output paths
        return my_output

    @property
    def work_path(self):
        return os.path.join(self.context.iter_path, '00.train')

    def executed(self):
        # it does the preparation of the training data.
        pass
