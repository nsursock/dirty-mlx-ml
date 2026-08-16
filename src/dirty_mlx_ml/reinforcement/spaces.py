class Box:
    def __init__(self, low, high, shape, dtype="float32"):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


class Discrete:
    def __init__(self, n):
        self.n = int(n)
        self.shape = ()
        self.dtype = "int32"
