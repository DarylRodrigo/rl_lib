from ..Memory import Memory
from ..Config import Config

def setup():
    mem = Memory(50, 2, )
    return mem

def should_init_memory():
    assert 3 == 3
