from aaa.models.layer_convertors.misc import __classinit


@__classinit
class LayerConvertor(object):
    @classmethod
    def _init__class(cls):
        cls._registry = { }

        return cls()

    def __call__(self, layer):
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @classmethod
    def _func_None(cls, layer):
        return layer
