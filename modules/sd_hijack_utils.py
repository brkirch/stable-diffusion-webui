import types

class ConditionalFunc:
    def __new__(cls, orig_func, sub_func, cond_func, *args):
        self = super(ConditionalFunc, cls).__new__(cls)
        self.__init__(orig_func, sub_func, cond_func, *args)
        return lambda *args, **kwargs: self(*args, **kwargs)
    def __init__(self, orig_func, sub_func, cond_func, *args):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
        if args:
            for key in args[0].keys():
                setattr(self, key, args[0][key])
    def __call__(self, *args, **kwargs):
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            return self.__orig_func(*args, **kwargs)

class GenericHijack:
    def __new__(cls, orig_object, remap_dict, *args):
        self = super(GenericHijack, cls).__new__(cls)
        self.__init__(orig_object, remap_dict, *args)
        return self
    def __init__(self, orig_object, remap_dict, *args):
        for key in remap_dict.keys():
            if not callable(remap_dict[key]):
                remap_dict[key] = ConditionalFunc(types.MethodType(remap_dict[key][0], self), remap_dict[key][1], remap_dict[key][2], *remap_dict[key][3:])
            else:
                remap_dict[key] = types.MethodType(remap_dict[key], self)
        if args:
            for key in args[0].keys():
                setattr(self, key, args[0][key])
        self.__orig_object = orig_object
        self.__remap_dict = remap_dict

    def __getattr__(self, item):
        if item in self.remap_dict.keys():
            return self.remap_dict[item]

        if hasattr(self.__orig_object, item):
            return getattr(self.__orig_object, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))
