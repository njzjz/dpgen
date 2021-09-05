class OPController(object):
    def __init__(
            self,
            op,
    ) -> None:
        self.op = op
        self.registered_i = {}
        self.registered_o = {}

    def get_input(self):
        return self.registered_i

    def get_output(self):
        return self.registered_o

    def register_input(func):
        @functools.wraps(func)
        def wrapper_register_i(self, pathobj, *args, **kwargs):
            func(self, *args, **kwargs)
            self.registered_i.add(pathobj)
        return wrapper_register_i

    def register_output(func):
        @functools.wraps(func)
        def wrapper_register_o(self, pathobj, *args, **kwargs):
            func(self, *args, **kwargs)
            self.registered_o.add(pathobj)
        return wrapper_register_o

    # wrap methods of Path, here are examples of register_input and register_output

    @register_input
    def read_text(pathobj, **kwargs):
        return pathobj.read_text(**kwargs)
        
    @register_output
    def mkdir(pathobj, **kwargs):
        pathobj.mkdir(**kwargs)
    
