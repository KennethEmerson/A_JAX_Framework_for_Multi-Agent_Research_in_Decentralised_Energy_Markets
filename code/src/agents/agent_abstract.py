""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the definition of the abstract Agent class
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
from train_data_objects import TrainState, _Loss, _Transition, _UpdateState


class Agent(ABC):
    """The abstract Agent class which defines the required interface for all concrete Agent child classes
    """  

    def __init__(self,nbr_of_updates:int) -> None:
        """ instantiates the object 

        Args:
            nbr_of_updates (int): the total number of updates during the complete training run. 
        """
        self.num_updates = nbr_of_updates



    @abstractmethod
    def create_init_train_state(self,observation_space_size:int,rng:jax.Array) -> TrainState:
        """ initializes the flax.training.train_state 
        """
        pass



    @abstractmethod
    def select_action(self,train_state:TrainState,observations:jax.Array,rng:jax.Array) -> Tuple[jax.Array,jax.Array,jax.Array,jax.Array]:
        """returns the selected action by the agent based on its trainstate and the given observations
        """
        pass



    @abstractmethod
    def update_networks(self,update_state:_UpdateState) -> Tuple[_UpdateState,_Loss]:
        """ updates the networks of the agent based on the given update state
        """
        pass



    @abstractmethod
    def update_networks_for_one_epoch(
            self,
            train_state:TrainState,
            last_obs_batch_per_agent:jax.Array,
            traj_batch:_Transition,
            rng:jax.Array) -> Tuple[_UpdateState,_Loss]:
        """updates the networks of the agent for one complete epoch

        """
        pass