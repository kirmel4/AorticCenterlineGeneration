from functools import partial

from aaa.utils.decorators import check_arguments

@partial(check_arguments, checkers=[])
def get_surface_ratio_rate(inner_surface, outer_surface):
    surface_ratio_rate = inner_surface / outer_surface

    return { 'surface ratio rate': float(surface_ratio_rate) }
