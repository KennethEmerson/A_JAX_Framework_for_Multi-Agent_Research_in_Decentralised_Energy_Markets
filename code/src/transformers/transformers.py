""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
------------------------------------------------------------------------------------------- 
file contains the implementation of the Transformer Class which is 
required as an interface between the 
environment and the agents
"""
import functools
from typing import Callable, NamedTuple

import environments.energymarket_env as _env
import globalMarket.globalmarket as _glo_ma
import jax
import jax.numpy as jnp
import transformers.transformer_helpers as _tf_helper
from transformers.transformers_config import (
    AgentActionPriceLimiter,
    AgentValueScaler,
)


class Statistics(NamedTuple):
    """ NamedTuple that contains all relevant statistics used by the transformer of a specific agent 
        These statistics can be used by the Transformer object to modify the agents observations and/or actions
        Note that all prosumer statistics are agent specific!
    """
    min_day_ahead_price:float
    q1_day_ahead_price:float
    q2_day_ahead_price:float
    q3_day_ahead_price:float
    max_day_ahead_price:float
    
    min_prosumer_demand:float
    q1_prosumer_demand:float
    q2_prosumer_demand:float
    q3_prosumer_demand:float
    max_prosumer_demand:float



###########################################
# SCALERS AND LIMITERS SELECTORS
###########################################

def _select_action_price_limiter(action_price_limiter:AgentActionPriceLimiter) -> Callable[[jax.Array,jax.Array,jax.Array],jax.Array]:
    """internal helper function that selects and returns the action price limiter function to use based on the given 
        AgentActionPriceLimiter Enum value
    """
    if action_price_limiter == AgentActionPriceLimiter.IGNORE:
        return _tf_helper.no_action_price_clipping
    elif action_price_limiter == AgentActionPriceLimiter.CLIP_TO_ZERO:
        return _tf_helper.clip_negative_action_price_to_zero
    elif action_price_limiter == AgentActionPriceLimiter.CLIP_TO_GLOB_MARKET:
        return _tf_helper.clip_action_price_to_globalmarket
    elif action_price_limiter == AgentActionPriceLimiter.TANH_NORMALIZE_TO_GLOB_MARKET:
        return _tf_helper.tanh_normalise_action_price_to_globalmarket
    elif action_price_limiter == AgentActionPriceLimiter.MAKE_ABS:
        return _tf_helper.make_action_price_absolute
    else:
        raise Exception(f"the action price limiter {action_price_limiter} is not implemented in the action transformer")



def _select_agent_value_scaler(
        value_scaler:AgentValueScaler,
        inverse_scaler:bool,
        min:float,
        quantile_1:float,
        quantile_2:float,
        quantile_3:float,
        max:float) -> Callable[[jax.Array],jax.Array]:
    """internal helper function that selects and returns a generic value scaler function to use based on the statical parameters 
        and the given AgentValueScaler Enum value
    """ 
    if value_scaler == AgentValueScaler.IGNORE:
        return _tf_helper.ignore_scaling()

    elif value_scaler == AgentValueScaler.ROBUST_SCALER:
        if inverse_scaler:
            return _tf_helper.inverse_robust_scaler(quantile_1,quantile_2,quantile_3)
        else:
            return _tf_helper.robust_scaler(quantile_1,quantile_2,quantile_3)
   
    elif value_scaler == AgentValueScaler.MINMAX_SCALER:
        if inverse_scaler:
            return _tf_helper.inverse_min_max_scaler(min,max)
        else:
            return _tf_helper.min_max_scaler(min,max)
    else:
        raise Exception(f"the agent_value_scaler {value_scaler} is not implemented in the action transformer")



###########################################
# TRANSFORMER CLASS
###########################################

class Transformer:
    """ implementation of the Transformer class for which the objects are used as an intermediate layer
        between the environment and the agent to which the transformer is assigned. A Transformer object will
        transform the observations from the environment datatype to an array which can be used as input for the agent.
        Actions selected by the agent will on their part be transformed from an array to the Action datatype as is used by 
        the environment. 

        beside both essential transformations, the Transformer can also be used to manipulate the actions and observations in order
        to for instance limit the agents action space to feasible values or extend manipulate or normalize the agents observations.
    """
    def __init__(self,
                agent_id:str,
                agent_index:int,
                clip_action_amounts_to_battery: bool,
                value_scaler:AgentValueScaler,
                action_price_limiter: AgentActionPriceLimiter,
                with_day_ahead_window: bool,
                nbr_data_sharing_agents:int, 
                globalmarket:_glo_ma.GlobalMarket,
                statistics:Statistics,
                 ) -> None:
        """_summary_

        Args:
            agent_id (str): the agent identifier (e.g. agent_00) as known by the agent linked to the transformer and the environment
            agent_index (int): the index value of the agent (e.g. 00) as known by the agent linked to the transformer and the environment
            clip_action_amounts_to_battery (bool): if true the agent action amounts will be clipped in such that their action 
                                                   can never result in a battery overflow or battery shortage 
            value_scaler (AgentValueScaler): the value scaler to use to normalise the prices and the energy amounts
            action_price_limiter (AgentActionPriceLimiter): the limiter function to use to limit the agents price 
            with_day_ahead_window (bool): if true the day ahead window provided by the global market will be made available to the agent
            nbr_data_sharing_agents (int): The number of agents (starting from agent_00) that share their 
                                           energy demand and battery level information with the following other agents. Note that this will 
                                           be determined by the index of the agent in the agent_list provided the environment. If this 
                                           value is 2 the first and second agent in that list will share their energy demand and battery 
                                           level. All others will just see zero values
            globalmarket (GlobalMarket): the Globalmarket Named Tuple containing all global market data
            statistics (Statistics): the statistics (min, qauntile 1,2,3 and max value of both the prosumer energy demand and global 
                                     market day ahead price) required for normalising observations and actions
        """
        self.agent_id=agent_id
        self.agent_index = agent_index
        self.with_day_ahead_window = with_day_ahead_window
        self.day_ahead_window_size = globalmarket.day_ahead_window_size
        self.nbr_data_sharing_agents = nbr_data_sharing_agents
        
        data_sharing = nbr_data_sharing_agents * 2 - 2 if nbr_data_sharing_agents > 0 else 0
        self.observation_space_size = 7 + self.day_ahead_window_size + data_sharing if self.with_day_ahead_window else 7 + data_sharing
        self.action_space_size = 2
        
        self.global_market_stats  = (
                    statistics.min_day_ahead_price,
                    statistics.q1_day_ahead_price,
                    statistics.q2_day_ahead_price,
                    statistics.q3_day_ahead_price,
                    statistics.max_day_ahead_price
                    )
        self.prosumer_stats = (
                    statistics.min_prosumer_demand,
                    statistics.q1_prosumer_demand,
                    statistics.q2_prosumer_demand,
                    statistics.q3_prosumer_demand,
                    statistics.max_prosumer_demand
                    )
        
        self.obs_globalmarket_price_scaler = _select_agent_value_scaler(value_scaler,False,*self.global_market_stats)
        self.action_price_inverse_scaler = _select_agent_value_scaler(value_scaler,True,*self.global_market_stats)
        
        self.obs_prosumer_energy_amount_scaler = _select_agent_value_scaler(value_scaler,False,*self.prosumer_stats)
        self.action_amount_inverse_scaler = _select_agent_value_scaler(value_scaler,True,*self.prosumer_stats)
        self.action_price_limiter = _select_action_price_limiter(action_price_limiter)

        if clip_action_amounts_to_battery:
            self.action_amount_limiter = _tf_helper.clip_action_amount_to_battery
        else:
            self.action_amount_limiter = _tf_helper.dont_clip_action_amount



    @functools.partial(jax.jit, static_argnums=(0,))
    def _create_data_sharing(self,energy_demand:jax.Array,battery_level:jax.Array,agent_index:int) -> jax.Array:
        """returns the array containing the energy demands and battery levels of all agents participating in the 
           sharing of data. All others will receive an array with zeros

        Args:
            energy_demand (jax.Array): the energy demands of all agents
            battery_level (jax.Array): the battery levels of all agents
            agent_index (int): the index of the agent associated with the transformer

        Returns:
            jax.Array: _description_
        """
        # create array for the participating agents
        def _get_true_values(arr:jax.Array,agent_index:int) -> jax.Array:  
            temp = jax.lax.dynamic_slice(arr,[0],[self.nbr_data_sharing_agents])
            return jnp.delete(temp,agent_index,axis=0,assume_unique_indices=True)

        # create array of zeros for nne participating agents
        def _get_zero_values(arr:jax.Array,agent_index:int) -> jax.Array:
            return jnp.zeros(self.nbr_data_sharing_agents-1,dtype=float)
        
        # determine which array to provide for the energy demand
        energy_demand_sharing = jax.lax.cond(
            agent_index < self.nbr_data_sharing_agents,
            _get_true_values,
            _get_zero_values,
            energy_demand,
            agent_index
        )
       
        # determine which array to provide for the battery levels
        battery_level_sharing = jax.lax.cond(
            agent_index < self.nbr_data_sharing_agents,
            _get_true_values,
            _get_zero_values,
            battery_level,
            agent_index
        )
       
        result =  jnp.concatenate((energy_demand_sharing,battery_level_sharing),axis=None)[::-1]
        return result



    @functools.partial(jax.jit, static_argnums=(0,))
    def transform_observations(
            self,
            state:_env.EnvironmentState,
            observations:_env.Observations, 
            ) -> jax.Array:
        """ takes in the agents observations as received from the environment, preprocesses them as 
            configured, and returns them as an array. 
             
        Args:
            state (_env.EnvironmentState): The environment state 
            observations (_env.Observations): the observations for the specific agent linked to the transformer and
                                              as provided by the environment

        Returns:
            jax.Array: the final observations usable for the agent
        """
        observation = observations[self.agent_id]                
    
        # rescale and or limit all observations
        temp_obs =  jnp.array([
                        2*jnp.array(observation["weeknumber"], dtype=float)/52 -1,
                        2*jnp.array(observation["hour_of_day"], dtype=float)/23 -1,
                        2*jnp.array(observation["day_of_week"], dtype=float)/6 -1,
                                            
                        self.obs_globalmarket_price_scaler(jnp.array(observation["time_of_use_price"], dtype=float)),
                        self.obs_globalmarket_price_scaler(jnp.array(observation["feed_in_price"], dtype=float)),
                        self.obs_prosumer_energy_amount_scaler(jnp.array(observation["energy_demand"], dtype=float)),
                        self.obs_prosumer_energy_amount_scaler(jnp.array(observation["battery_level"], dtype=float)),
                    ]) 

        # add day ahead window if applicable     
        obs = jnp.concatenate([temp_obs,jnp.array(observation["day_ahead_window"]).T],axis=0) if self.with_day_ahead_window else temp_obs

        energy_demands = jnp.array([self.obs_prosumer_energy_amount_scaler(state.energy_demand[agent_id]) for agent_id in observations.keys()]) 
        battery_levels = jnp.array([self.obs_prosumer_energy_amount_scaler(state.battery_level[agent_id]) for agent_id in observations.keys()]) 
               
        # select final observation based on nbr_sharing > 0, if so add data sharing to observations else do nothing
        obs_final = jnp.concatenate(
            [obs,self._create_data_sharing(energy_demands,battery_levels,self.agent_index)],
             axis=None) if self.nbr_data_sharing_agents > 0 else obs
            
        return obs_final 


    def transform_actions(
            self,
            state:_env.EnvironmentState,
            action:jax.Array,
            ) -> _env.Action:
        """takes in the agent's actions, preprocesses them as 
            configured, and returns them as a dict as expected by the environment. 

        Args:
            state (_env.EnvironmentState): the environment state
            action (jax.Array): the array containing the actions of the agent

        Returns:
            _env.Action: a TypedDict containing all relevant action data from the agent to the environment
        """
        raw_price = jnp.array(action[0],ndmin=1)
        raw_amount = jnp.array(action[1],ndmin=1)
        
        price = jnp.array(
            self.action_price_limiter(
                    state.time_of_use_price,
                    state.feed_in_price,
                    self.action_price_inverse_scaler(jnp.array(action[0],ndmin=1))
                    )
            )

        amount = jnp.array(self.action_amount_limiter(
                    max_battery_peak_Wh = state.max_battery_peak_Wh[self.agent_id],
                    max_battery_capacity_Wh = state.max_battery_capacity_Wh[self.agent_id],
                    battery_level_Wh = state.battery_level[self.agent_id],
                    energy_bought_Wh = self.action_amount_inverse_scaler(jnp.array(action[1],ndmin=1)),
                    energy_demand_Wh = state.energy_demand[self.agent_id]
                    ))

        action_dict = _env.Action({
                "raw_price":raw_price,
                "raw_amount":raw_amount,
                "price": price,
                "amount": amount
                })
        return action_dict
