""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains internal datatypes used during training in train.py 
"""

from typing import Any, Dict, NamedTuple, Union

import jax
import jax.numpy as jnp
from environments.energymarket_env import EnvironmentState, Observation
from flax.training.train_state import TrainState


class _TransitionAllAgents(NamedTuple):
    done: dict[str,jnp.ndarray]
    action: dict[str,jnp.ndarray]
    value:  Any
    reward: dict[str,jnp.ndarray]
    log_prob: dict[str,jnp.ndarray]
    obs: dict[str,jnp.ndarray]
    info: jnp.ndarray


class _Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value:  Any
    reward: jnp.ndarray
    log_prob: jnp.ndarray 
    obs: jnp.ndarray
    info: jnp.ndarray


class _UpdateState(NamedTuple):
    train_state:TrainState
    traj_batch:_Transition
    advantages:jax.Array
    targets:Any
    rng:Any


class _RunnerState(NamedTuple):
    train_states:Union[TrainState,Dict[str,TrainState]]
    env_state:EnvironmentState
    last_obs:Union[Dict[str,Observation],Any]
    rng:Any


class _BatchInfo(NamedTuple):
    traj_batch:jax.Array
    advantages:jax.Array
    targets:jax.Array


class _Loss(NamedTuple):
    value_loss:jax.Array
    loss_actor:jax.Array
    entropy:jax.Array