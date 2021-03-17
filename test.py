from functools import wraps


def add_prefix(fun):
    @wraps(fun)
    def decorated(self):
        pre_fix = self.pre_fix
        input_str = pre_fix + fun(self, 'hello')
        print(input_str)
        return input_str

    return decorated


class Test:
    def __init__(self):
        self.pre_fix = 'decorated_'

    @add_prefix
    def add_postfix(self, input_str):
        return input_str + '_postfix'

class Test2(Test):
    def __init__(self):
        super().__init__()

    def add_postfix(self, input_str):
        return input_str + '_postfix2'

t = Test2()
t.add_postfix('test')
