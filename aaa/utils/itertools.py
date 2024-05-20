from itertools import islice, tee


def take(iterable, n):
    return list(islice(iterable, n))

def unchain(iterable, n):
    iterable = iter(iterable)

    while True:
        result = take(iterable, n)
        if result:
            yield result
        else:
            break

def pairwise(iterable):
    first, second = tee(iterable)
    next(second, None)
    yield from zip(first, second)

def triplewise(iterable):
    first, second, third = tee(iterable, 3)
    next(second, None)
    next(third, None)
    next(third, None)

    yield from zip(first, second, third)