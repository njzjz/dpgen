import os
from pathlib import Path

# format constants
iteration_format = 'iter.%06d'
iteration_pattern = 'iter.[0-9][0-9][0-9][0-9][0-9][0-9]'
task_format = 'task.%06d'
task_pattern = 'task.[0-9][0-9][0-9][0-9][0-9][0-9]'
iterdata_format = 'data.%06d'
iterdata_pattern = 'data.[0-9][0-9][0-9][0-9][0-9][0-9]'
train_format = 'train.%03d'
train_pattern = 'train.[0-9][0-9][0-9]'

# steps in iterations
step_train = '00.train'
step_model_devi = '01.model_devi'
step_fp = '02.fp'

class Context(object):
    r"""The context of DP-GEN events.

    Parameters
    ----------
    local_path: str
                The path on local machine where the event happens
    """
    def __init__(self,
                 dpgen_path : str,
    ) -> None:
        # for example /path/to/dpgen
        self._dpgen_path = Path(dpgen_path)

    @property
    def dpgen_path(self):
        return self._dpgen_path


class IterationContext(Context):
    def __init__(self,
                 dpgen_path : str,
                 iteration : int = 0,
    ) -> None:
        super().__init__(dpgen_path)
        # for example iter.000002
        self._iter_path = Path(iteration_format%iteration)
        # for example iter.000001
        if iteration > 0 :
            self._prev_iter_path = Path(iteration_format%(iteration-1))
        else:
            self._prev_iter_path = None
        # for example iter.000003
        self._next_iter_path = Path(iteration_format%(iteration+1))
        # for example [iter.000000, iter.000001]
        self._all_prev_iter = [Path(iteration_format % ii) for ii in range(iteration)]

    @property
    def iter_path(self):
        return self._iter_path

    @property
    def prev_iter_path(self):
        return self._prev_iter_path

    @property
    def next_iter_path(self):
        return self._next_iter_path

    @property
    def all_prev_iter(self):
        return self._all_prev_iter

