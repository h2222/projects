
import params
from .create_generators import create_generator

def gen():
    g = create_generator(params=params, mode='Train')
    # to avoid nested parallel
    for instance in g:
        yield instance


print(next(gen()))