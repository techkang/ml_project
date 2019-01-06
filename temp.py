import numpy as np


class Variable(np.ndarray):
    """This class is inherit from np.ndarray to save value and gradient at the
    same time. The class of data must be np.ndarray.

    """

    def __new__(cls, array):
        return super(Variable, cls).__new__(cls, array.shape)

    def __init__(self, array):
        super(Variable, self).__init__()
        self[:] = array.copy()
        self.grad = np.zeros(self.shape)

    def __call__(self, *args, **kwargs):
        print('You are calling me!')
        print(args)
a = np.array([1.1, 2.2])
b = Variable(a)
print(b.grad)
print(type(b+a))