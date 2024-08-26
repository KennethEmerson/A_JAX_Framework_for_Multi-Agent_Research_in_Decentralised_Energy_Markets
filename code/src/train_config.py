""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the data class implementation containing all training specific parameters
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """
    dataclass encapsulating the parameters that define the training 
    process 

    Args:
        SELF_PLAY (bool): if true agent will engage in self-play 
        SEED (int): random seed used for reproducibility in training. 
        NUM_STEPS (int): The number of steps to take per update, determining how many 
                         interactions with the environment occur before the agent's 
                         parameters are updated.
        EPOCH_STEPS (int): number of steps to take before the environment is considered 
                           done for the current epoch
        TOTAL_TIMESTEPS (int): The total number of timesteps to take in the complete training run
    """
    SELF_PLAY:bool
    SEED: int
    NUM_STEPS : int # number of steps per update
    EPOCH_STEPS : int # number of steps before env goes to done
    TOTAL_TIMESTEPS : int # total timesteps to take in complete training run (over multiple epochs)