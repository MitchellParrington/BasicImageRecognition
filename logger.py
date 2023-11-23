class Logger:
    def __init__(self, v=0):
        self.v = v

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def log(self, *args, a=2, **kwargs):
        if a <= self.v:
            print(*args, **kwargs)
