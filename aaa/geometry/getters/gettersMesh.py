from functools import partial

from aaa.utils.decorators import check_arguments

@partial(check_arguments, checkers=[])
def get_surface_square(surface, spacing):
    surface_square = surface.area * spacing**2

    return { 'surface square': float(surface_square),
             'square unit': 'mm^2' }
