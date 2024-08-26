""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains an abstract dataclass implementation preventing concrete child classes to be instantiated without
the abstract dataclass parameters

based on: 
https://stackoverflow.com/questions/60590442/abstract-dataclass-without-abstract-methods-in-python-prohibit-instantiation
"""
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class AbstractDataclass(ABC): 
    """ abstract dataclass implementation preventing concrete child classes to be instantiated without
        the abstract dataclass parameters
    """
    def __new__(cls, *args, **kwargs): 
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass: 
            raise TypeError(f"Cannot instantiate an abstract data class {cls.__name__}") 
        return super().__new__(cls)