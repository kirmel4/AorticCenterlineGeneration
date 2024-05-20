
def __get_args_dict(func, args, kwargs):
    args_count = func.__code__.co_argcount
    args_names = func.__code__.co_varnames[:args_count]

    if func.__defaults__:
        args_defaults_count = len(func.__defaults__)
        args_defaults_names = args_names[-args_defaults_count:]

        args_dict = { **dict(zip(args_defaults_names, func.__defaults__)),
                      **dict(zip(args_names, args)) }
    else:
        args_dict = dict(zip(args_names, args))

    if func.__code__.co_kwonlyargcount:
        kwargs_dict = {**func.__kwdefaults__, **kwargs}
    else:
        kwargs_dict = kwargs

    return {**args_dict, **kwargs_dict}

def check_arguments(func, checkers=[]):
    def wrapper_func(*args, **kwargs):
        for checker in checkers:
            checker(__get_args_dict(func, args, kwargs))

        return func(*args, **kwargs)
    return wrapper_func
