""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains all dataclass definitions that contain all the required parameters to create a 
specific Agent class implementation object
"""

from dataclasses import dataclass
from typing import List

from abstract_dataclass import AbstractDataclass


@dataclass(frozen=True)
class ActorCriticNeurNetConfig:
    """dataclass containing the layer sizes for the actor and critic layer
    """
    ACTOR_LAYERS : List[int] # list of integers corresponding to the layer sizes of the actor; one element in the list corresponds to one layer
    CRITIC_LAYERS : List[int] # list of integers corresponding to the layer sizes of the critic; one element in the list corresponds to one layer



@dataclass(frozen=True)
class AgentConfig(AbstractDataclass):
    """abstract configuration dataclass for an agent
    """
    ID:str



@dataclass(frozen=True)
class RuleBasedAgentConfig(AgentConfig):
    """configuration dataclass containing all parameters required to create the RuleBasedAgent class object
    """
    USE_DAY_AHEAD_WINDOW_IF_AVAILABLE:bool  # currently not implemented in the RuleBasedAgent
    USE_DATA_SHARING_IF_AVAILABLE:bool      # currently not implemented in the RuleBasedAgent



@dataclass(frozen=True)
class PpoAgentConfig(AgentConfig):
    """configuration dataclass containing all parameters required to create a PpoAgent object
    """
    NN_MODEL:ActorCriticNeurNetConfig
    ACTIVATION : str # activation function to use in NN if not "relu" then tanh is used ex. "relu"
    MAX_GRAD_NORM : float # gradient clipping to prevent exploding gradients when optimizing
    GAMMA : float # reward discount factor
    GAE_LAMBDA : float # the General Advantage Estimation 
    CLIP_EPS : float # CLIP epsilon that clips the objective function (in PPO paper set to 0.2)
    VF_COEF : float # value function loss coefficient in the objective function  
    ENT_COEF : float # the entropy bonus in the objective function
    LR : float # initial Learning rate
    ANNEAL_LR: bool # if set to true Learning rate will decrease linearly
    UPDATE_EPOCHS : int  # impacts the LR decay (higher is slower)
    NUM_MINIBATCHES : int # nbr of mini batches to backprop on for each set of NUM_STEPS + impacts the LR decay (higher is slower) 
