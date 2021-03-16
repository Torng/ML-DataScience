
from collections import namedtuple
from typing import NamedTuple
import numpy as np




class TestClass(NamedTuple):
    test1:str
    test2:list
    test3:int

new_named_class = TestClass('Hawk',[1,2,3,4],666)

print(new_named_class.test1)
print(new_named_class.test2)
print(new_named_class.test3)