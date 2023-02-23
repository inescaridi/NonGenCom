
class BaseScore(object):
    
    def __call__(self, **kw):
        raise NotImplementedError("Please implement in sub classes")
