""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the implementation of a rule based agent class
The class interface is as such that these agents can be used in experiments in combination
with other agent implementations such as PPO. 
The agent uses a simple rule in such that its energy consumption is positive it places an ask
offer for the full energy demand and with the global market time of use price as asking price.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from agents.agent_abstract import Agent
from agents.agent_config import RuleBasedAgentConfig
from flax.training.train_state import TrainState
from train_config import TrainConfig
from train_data_objects import _Loss, _Transition, _UpdateState
from transformers.transformers_config import TransformerConfig


class RuleBasedAgent(Agent):
    """ Agent Class implementing a rule based agent
    """ 
    def __init__(
            self,
            agent_config:RuleBasedAgentConfig,
            train_config:TrainConfig,
            transformer_config:TransformerConfig,
            action_space_size:int) -> None:
        """ instantiates the Agent object

        Args:
            agent_config (RuleBasedAgentConfig): a AgentConfig dataclass for a rule based agent containing all agent specific parameters
            train_config (TrainConfig): the TrainConfig dataclass containing all training specific parameters
            transformer_config (TransformerConfig): the TransformerConfig dataclass containing all transformer specific parameters
            action_space_size (int): the size of the action space
        """

        self.agent_config = agent_config
        self.action_space_size = action_space_size
        self.day_ahead_window = transformer_config.WITH_DAY_AHEAD_WINDOW
        self.nbr_of_sharing_agents = transformer_config.DATA_SHARING_AGENTS
        super().__init__(train_config.TOTAL_TIMESTEPS // train_config.NUM_STEPS)



    ##############################################################################
    # MAIN METHODS
    ##############################################################################  

    def create_init_train_state(self,observation_space_size:int,rng:jax.Array) -> TrainState:
        """ initializes the flax.training.train_state of the agent

        Args:
            observation_space_size (int): the size of the observation space
            rng (jax.Array): a JAX random number

        Returns:
            TrainState: the flax.training.train_state of the agent
        """

        def _calc_actions(train_state:TrainState, observations:jax.Array) -> Tuple[jax.Array,jax.Array]:
            """ the actual rule to select the action.
                The agent uses a simple rule in such that its energy consumption is positive it places an ask
                offer for the full energy demand and with the global market time of use price as asking price.
            """
                       
            weeknumber = observations[0]
            hour_of_day = observations[1]
            day_of_week = observations[2]
            time_of_use_price = observations[3]
            feed_in_price = observations[4]
            energy_demand = observations[5]
            battery_level = observations[6]

            
            action_price = jnp.where(energy_demand > 0,
                                    time_of_use_price,
                                    jnp.where(energy_demand < 0,
                                                feed_in_price,
                                                0
                                                )
                                    )

            actions = jnp.array([action_price,energy_demand])
            return jnp.zeros(2), actions

        self.observation_space_size = observation_space_size 

        # creating the Trainstate is only implemented to comply with the API of the PPO agent
        # but is however never used
        network_params = jnp.zeros((self.observation_space_size))
        tx = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(0.01, eps=1e-5))
        train_state = TrainState.create(
            #train_config = self.train_config,
            apply_fn=_calc_actions,
            params=network_params,
            tx= tx,
        )
        return train_state



    @functools.partial(jax.jit, static_argnums=(0,))
    def select_action(self,train_state,observations:jax.Array,rng:jax.Array):
        """returns the selected action by the agent based on the given observations

        Args:
            train_state (TrainState): the current flax.training.train_state of the agent (not used by the rule based agent)
            observations (jax.Array): the observations on which to select the new action
            rng (jax.Array): a JAX random number

        Returns:
            Tuple[jax.Array,jax.Array,jax.Array,jax.Array]: a tuple containing the following jax arrays:
                                                            policy function pi, critic value, action, action log_probability.
                                                            The policy function pi, critic value, action log_probability are however
                                                            nonsensical in the context of the rule based agent and are therefore filled
                                                            with zeros 
        """
        pi, value = train_state.apply_fn(train_state.params, observations)   
        action = value
        log_prob = jnp.zeros(2)

        return pi, value, action, log_prob


   
    @functools.partial(jax.jit, static_argnums=(0,))
    def update_networks(self,update_state):
        """ nonsensical dummy function in case of the rule based agent

        Args:
            update_state (_UpdateState): The actual update state of the agent, 
                                         containing the actual trainstate, the trajectory batch, targets...

        Returns:
            Tuple[_UpdateState,_Loss]: returns the input update state and a array of zeros
        """        
        total_loss = jnp.zeros(1)
        return update_state, total_loss
    

    
    @functools.partial(jax.jit, static_argnums=(0,))
    def update_networks_for_one_epoch(
            self,
            train_state:TrainState,
            last_obs_batch_per_agent:jax.Array,
            traj_batch:_Transition,
            rng:jax.Array) -> Tuple[_UpdateState,_Loss]:
        """ nonsensical dummy function in case of the rule based agent

        Args:
            train_state (TrainState): the flax.training.train_state of the agent
            last_obs_batch_per_agent (jax.Array): a batch containing the last observations used for the training of the NN
            traj_batch (_Transition): a batch containing the observations trajectory used for the training of the NN
            rng (jax.Array): a JAX random number

        Returns:
            Tuple[_UpdateState,_Loss]: returns the input update state and a zero filled total loss
        """
        
        advantages = jnp.zeros(1)
        targets = jnp.zeros(1)
        update_state = _UpdateState(train_state, traj_batch, advantages, targets, rng)
        loss = _Loss(jnp.zeros(0),jnp.zeros(1),jnp.zeros(1))
        return update_state, loss