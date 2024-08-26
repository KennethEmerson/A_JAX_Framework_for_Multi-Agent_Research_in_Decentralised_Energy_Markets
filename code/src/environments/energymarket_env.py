""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
This file contains the energy market environment definition and complies to the
interface standard as used in the JaxMARL library
For more info on jaxMARL: https://github.com/FLAIROx/JaxMARL
"""

import functools
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Tuple, TypedDict, Union

import chex  #type:ignore
import environments.spaces as spaces
import globalMarket.globalmarket as gloma
import jax  #type:ignore
import jax.numpy as jnp
import ledger.ledger as ledger
from auction import doubleAuction
from environments.environment import MultiAgentEnv
from prosumer.prosumer import Prosumer

#####################################################################################################################
# CONSTANTS
#####################################################################################################################

MIN_MONTH = 0
MAX_MONTH = 12
DTYPE_MONTH = jnp.int32

MIN_WEEKNUMBER = 0
MAX_WEEKNUMBER = 52
DTYPE_WEEKNUMBER = jnp.int32

MIN_HOUR_OF_DAY = 0
MAX_HOUR_OF_DAY = 24
DTYPE_HOUR_OF_DAY = jnp.int32

MIN_DAY_OF_WEEK = 0
MAX_DAY_OF_WEEK = 7
DTYPE_DAY_OF_WEEK = jnp.int32

MIN_ACTION_SPACE = 0.0
MAX_ACTION_SPACE = 1000.

#####################################################################################################################
# SPACES
#####################################################################################################################

ObservationSpace = spaces.Dict({
    "weeknumber":spaces.Box(low=MIN_WEEKNUMBER, high= MAX_WEEKNUMBER, shape=(1,), dtype= DTYPE_WEEKNUMBER), # type: ignore
    "hour_of_day":spaces.Box(low=MIN_HOUR_OF_DAY, high= MAX_HOUR_OF_DAY, shape=(1,), dtype= DTYPE_HOUR_OF_DAY), # type: ignore
    "day_of_week":spaces.Box(low=MIN_DAY_OF_WEEK, high=MAX_DAY_OF_WEEK, shape=(1,), dtype=DTYPE_DAY_OF_WEEK), # type: ignore
    "time_of_use_price":spaces.Box(low= -jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "feed_in_price":spaces.Box(low= -jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "day_ahead_window":spaces.Box(low= -jnp.inf, high=jnp.inf, shape=(24,), dtype=jnp.float32), # type: ignore
    "energy_demand":spaces.Box(low= -jnp.inf, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "max_energy_production_Wh":spaces.Box(low=0.0, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "battery_level":spaces.Box(low=0.0, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "max_battery_capacity_Wh":spaces.Box(low=0.0, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
    "max_battery_peak_Wh":spaces.Box(low=0.0, high=jnp.inf, shape=(1,), dtype=jnp.float32), # type: ignore
        })

ActionSpace = spaces.Dict({
    "raw_price" :spaces.Box(low=MIN_ACTION_SPACE, high= MAX_ACTION_SPACE, shape=(1,),dtype=jnp.float32), # type: ignore
    "raw_amount":spaces.Box(low=MIN_ACTION_SPACE, high= MAX_ACTION_SPACE, shape=(1,),dtype=jnp.float32), # type: ignore
    "price" :spaces.Box(low=MIN_ACTION_SPACE, high= MAX_ACTION_SPACE, shape=(1,),dtype=jnp.float32), # type: ignore
    "amount":spaces.Box(low=MIN_ACTION_SPACE, high= MAX_ACTION_SPACE, shape=(1,),dtype=jnp.float32) # type: ignore
    })

RewardSpace = spaces.Box(low=0, high=1000, shape=(1,), dtype=jnp.float32) # type:ignore


#####################################################################################################################
# TYPES
#####################################################################################################################

@chex.dataclass
class EnvironmentState:
    """
    data class representing the state of the environment.

    Args:
        epoch (int): The current epoch or time period in the simulation.
        timestep (int): The specific timestep within the current epoch.
        time_of_use_price (jax.Array): An array representing the time-of-use pricing of the global market.
        feed_in_price (jax.Array): An array representing the feed-in tariffs of the global market.
        day_ahead_window (jax.Array): An array representing the prices for the next day ahead in the global market.
        energy_demand (Dict[str, jax.Array]): A dictionary mapping agents to their respective energy demand values.
        max_energy_production_Wh (Dict[str, jax.Array]): A dictionary mapping agents to their maximum energy production capacities in Wh.
        battery_level (Dict[str, jax.Array]): A dictionary mapping agents to their current battery energy levels in Wh.
        max_battery_capacity_Wh (Dict[str, jax.Array]): A dictionary mapping agents to their maximum battery energy storage capacities in Wh.
        max_battery_peak_Wh (Dict[str, jax.Array]): A dictionary mapping agents to their maximum battery energy transfer rate in Wh.

    Notes:
        - All prices must be expressed in the same currency.
        - All energy and battery values must be represented in the same unit (e.g., watt-hours, Wh).
    """
    epoch:int
    timestep:int
    time_of_use_price:jax.Array
    feed_in_price:jax.Array
    day_ahead_window:jax.Array 
    energy_demand:Dict[str,jax.Array] 
    max_energy_production_Wh:Dict[str,jax.Array] 
    battery_level:Dict[str,jax.Array]
    max_battery_capacity_Wh:Dict[str,jax.Array]
    max_battery_peak_Wh:Dict[str,jax.Array]


Observation = OrderedDict[str, jax.Array]
Observations = Dict[str, Observation ]


class Action(TypedDict):
    """
    A TypedDict representing an agents action/offer.

    Args:
        raw_price (jax.Array): An array representing the unprocessed price associated with the action/offer.
        raw_amount (jax.Array): An array representing the unprocessed amount of energy associated with the action/offer.
        price (jax.Array): An array representing the transformed price for the action/offer.
        amount (jax.Array): An array representing the transformed amount of energy in the action/offer.
    """
    raw_price: jax.Array
    raw_amount:jax.Array
    price: jax.Array
    amount:jax.Array


Actions = Dict[str, Action ]
Reward = jax.Array
Rewards = Dict[str, Reward ]
Infos = dict 


class _BatteryValues(NamedTuple):
    """
    represents the battery level values as a consequence of trading on the internal and global market .

    Args:
        battery_level (jax.Array): represents the current energy level of the battery in Wh after processing 
                                   the trading on both internal and global market.
        overflow (jax.Array): represents the amount of energy that exceeds the battery's maximum capacity after 
                              processing the trading on both internal and global market.
        shortage (jax.Array): represents the amount of energy that is lacking to meet the actual agents energy 
                              demand after processing the trading on both internal and global market.
    """
    battery_level:jax.Array
    overflow: jax.Array
    shortage:jax.Array


#####################################################################################################################
# helper function for calculating the battery levels 
#####################################################################################################################

@functools.partial(jax.jit,static_argnames=("agent_ids"))
def _calculate_battery_level_and_excesses(
    state:EnvironmentState,
    agent_ids:Tuple[str, ...],
    offer_amounts:jax.Array) -> _BatteryValues:
    """
    Calculate the updated battery levels and overflow and shortage of energy based on the environment state 
    (and thus actual battery levels), the offer amounts and the constraints of battery capacity and peak (dis)charge rates.

    Args:
        state (EnvironmentState): The current state of the environment.
        agent_ids (Tuple[str, ...]): A tuple of agent identifiers for which the battery levels and excesses are to be calculated.
        offer_amounts (jax.Array): An array representing the amounts of energy offered by the agents.

    Returns:
        _BatteryValues: named tuple containing the new battery levels, battery overflow and shortage 
    """
            
    energy_demand = jnp.ravel(jnp.array([state.energy_demand[agent_id] for agent_id in agent_ids]))
    battery_level = jnp.ravel(jnp.array([state.battery_level[agent_id] for agent_id in agent_ids]))
    max_battery_capacity_Wh = jnp.ravel(jnp.array([state.max_battery_capacity_Wh[agent_id] for agent_id in agent_ids]))
    max_battery_peak_Wh = jnp.ravel(jnp.array([state.max_battery_peak_Wh[agent_id] for agent_id in agent_ids]))


    # calculate new theoretical battery load without considering max and min
    battery_delta_unclipped = offer_amounts - energy_demand
    
    # calculate how much excess energy is available that cannot be loaded in the battery due to peak
    battery_delta_overflow  = jnp.where(max_battery_peak_Wh < battery_delta_unclipped, battery_delta_unclipped - max_battery_peak_Wh,0) 
    
    # calculate how much additional energy is required that cannot be provided due to battery peak
    battery_delta_shortage = jnp.abs(jnp.where(- max_battery_peak_Wh > battery_delta_unclipped, max_battery_peak_Wh + battery_delta_unclipped,0))
    
    # clip the total battery charge discharge 
    battery_delta_pos_clipped = jnp.where(battery_delta_overflow > 0, max_battery_peak_Wh, battery_delta_unclipped)
    battery_delta_clipped = jnp.where(battery_delta_shortage > 0, - max_battery_peak_Wh, battery_delta_pos_clipped)
    

    # calculate new theoretical battery level without considering max and min by using the battery delta clipped
    new_battery_level_unclipped = battery_level + battery_delta_clipped
    
    # calculate how much excess energy is available that cannot be stored in the battery
    battery_level_overflow = jnp.where(max_battery_capacity_Wh < new_battery_level_unclipped, new_battery_level_unclipped - max_battery_capacity_Wh,0) 
    
    # calculate how much additional energy is required that cannot be provided by the battery
    battery_level_shortage = jnp.abs(jnp.where(new_battery_level_unclipped < 0,new_battery_level_unclipped,0))
    
    # clip battery levels to max battery capacity
    new_battery_level_with_upper_clip = jnp.where(battery_level_overflow > 0, max_battery_capacity_Wh, new_battery_level_unclipped)
    
    # clip battery levels to min battery capacity = 0
    new_battery_level_clipped = jnp.where(battery_level_shortage > 0,0, new_battery_level_with_upper_clip)

    return _BatteryValues(
            battery_level = new_battery_level_clipped,
            overflow = battery_level_overflow + battery_delta_overflow,
            shortage = battery_level_shortage + battery_delta_shortage
    )



#####################################################################################################################
# MAIN ENVIRONMENT DEFINITION
#####################################################################################################################

class DoubleAuctionEnv(MultiAgentEnv):
    """
    Jax Compatible version of an double Auction environment which complies to the
    interface standard as used in the JaxMARL library
    """
    def __init__(
            self,
            global_market: gloma.GlobalMarket,
            agent_prosumers:Union[List[Prosumer],Tuple[Prosumer]],
            epoch_nbr_of_steps:int
            ) -> None:
        """ instantiates the environment object
        Args:
            global_market (GlobalMarket): the GlobalMarket class instantiated object
            agent_prosumers (Union[List[Prosumer],Tuple[Prosumer]]): A list or tuple of Prosumer class instantiated objects
            epoch_nbr_of_steps (int): number of steps in each epoch 
        """

        num_agents = len(agent_prosumers)
        super().__init__(num_agents)
        self.__name__ = "double_action"
        
        self.agent_ids = tuple(f"agent_{i:02d}" for i in range(self.num_agents))
        self.agent_ids_to_index = {f"agent_{i:02d}":i for i in range(self.num_agents)}
        self.index_to_agent_ids = {i:f"agent_{i:02d}" for i in range(self.num_agents)}
        self.agent_prosumers = {f"agent_{i:02d}": prosumer for i, prosumer in enumerate(agent_prosumers)}
        self.global_market= global_market
        
        self.observation_spaces = dict(zip(self.agent_ids, [ObservationSpace] * self.num_agents))
        self.action_spaces = dict(zip(self.agent_ids, [ActionSpace] * self.num_agents))
        self.reward_spaces = dict(zip(self.agent_ids, [RewardSpace] * self.num_agents))
        
        # The number of steps per epoch is limited by the minimal amount of samples in all agent_prosumers and the global market
        # This to prevent one of the prosumers or the global market running out of samples
        self.max_steps = min(len(min(agent_prosumers, key=lambda obj: len(obj.energy_consumption_Wh)).energy_consumption_Wh),global_market.nbr_of_samples)
        self.epoch_nbr_of_steps = min(self.max_steps,epoch_nbr_of_steps)

    

    @property
    def name(self) -> str:
        """ The Environment name.
        """
        return type(self).__name__



    @property
    def agent_classes(self) -> Dict:
        """Returns a dictionary with agent classes, used in environments with heterogenous agents.
        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError



    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id:str = "") -> ObservationSpace: # type: ignore
        """ returns the observation space for the given agent identifier
        Args:
            agent_id (str, optional): the agent identifier for which to retrieve the observation space. Defaults to "".
        Returns:
            RewardSpace: the observation space for the given agent  
        Notes:
            Because the observation space is identical to all agents. the agent_id is unused
        """
        return ObservationSpace[self.agent_ids[0]]



    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id:str = "") -> ActionSpace: # type: ignore
        """ returns the action space for the given agent identifier
        Args:
            agent_id (str, optional): the agent identifier for which to retrieve the action space. Defaults to "".
        Returns:
            RewardSpace: the action space for the given agent 
        Notes:
            Because the action space is identical to all agents. the agent_id is unused
        """
        return ActionSpace[self.agent_ids[0]]
    


    @functools.lru_cache(maxsize=None)
    def reward_space(self, agent_id:str = "") -> RewardSpace: # type: ignore
        """ returns the reward space for the given agent identifier
        Args:
            agent_id (str, optional): the agent identifier for which to retrieve the reward space. Defaults to "".
        Returns:
            RewardSpace: the reward space for the given agent  
        Notes:
            Because the reward space is identical to all agents. the agent_id is unused
        """
     
        return RewardSpace



    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvironmentState) -> Observations:
        """ returns the individual observations for each agent
        Args:
            state (State): the environment state on which the observations are based.
        Returns:
            Observations: Dictionary mapping agent identifiers to an ordered dict of observation values
        """
        def _create_observation(agent_id:str,state:EnvironmentState,prosumers:Dict[str,Prosumer])-> Observation:
            timestep = state.timestep
            weeknumber = prosumers[agent_id].weeknumbers[timestep]
            hour_of_day = prosumers[agent_id].hours[timestep]
            day_of_week = prosumers[agent_id].weekdays[timestep]
            time_of_use_price = state.time_of_use_price
            feed_in_price = state.feed_in_price
            energy_demand = state.energy_demand[agent_id]
            max_energy_production_Wh = state.max_energy_production_Wh[agent_id]
            battery_level = state.battery_level[agent_id]
            max_battery_capacity_Wh  = state.max_battery_capacity_Wh[agent_id]
            max_battery_peak_Wh = state.max_battery_peak_Wh[agent_id]

            return OrderedDict({
                "weeknumber":weeknumber,
                "hour_of_day":hour_of_day,
                "day_of_week":day_of_week,
                "time_of_use_price":time_of_use_price,
                "feed_in_price":feed_in_price,
                "day_ahead_window": state.day_ahead_window,
                "energy_demand":energy_demand,
                "max_energy_production_Wh":max_energy_production_Wh,
                "battery_level":battery_level,
                "max_battery_capacity_Wh": max_battery_capacity_Wh,
                "max_battery_peak_Wh": max_battery_peak_Wh
                })
        
        return {agent_id: _create_observation(agent_id,state,self.agent_prosumers) for agent_id in self.agent_ids}



    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey,epoch:int=0) -> Tuple[Observations, EnvironmentState]:
        """ resets the environment and returns an initial state and the observations for each agent.
        Args:
            key (chex.PRNGKey): random key
            epoch (int,optional): the epoch index to use after the rest, defaults to 0.

        Returns:
            Tuple[Dict[str, chex.Array], State]: tuple containing 
                                                    1) a dict mapping the agent identifiers to their initial observations as value and 
                                                    2) the state.
        """
        reset_state = EnvironmentState(
                epoch = epoch,
                timestep = 0,
                time_of_use_price = gloma.calc_variable_time_of_use_price(self.global_market,0),
                feed_in_price = gloma.calc_variable_feed_in_price(self.global_market,0),
                day_ahead_window = gloma.get_day_ahead_window(self.global_market,0),
                energy_demand = {agent_id:self.agent_prosumers[agent_id].energy_consumption_Wh[0] for agent_id in self.agent_ids},
                max_energy_production_Wh = {agent_id:self.agent_prosumers[agent_id].max_energy_production_Wh for agent_id in self.agent_ids},
                battery_level = {agent_id:jnp.array(0.) for agent_id in self.agent_ids}, 
                max_battery_capacity_Wh = {agent_id:self.agent_prosumers[agent_id].max_battery_capacity_Wh[0] for agent_id in self.agent_ids}, 
                max_battery_peak_Wh = {agent_id:self.agent_prosumers[agent_id].max_battery_peak_Wh[0] for agent_id in self.agent_ids} 
            ) 

        observations = self.get_obs(reset_state)
        return observations, reset_state



    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvironmentState,
        actions: Actions,
    ) -> Tuple[Observations, EnvironmentState, Rewards, Dict[str, bool], Infos]:
        """ performs a step in the environment and performs a rest if environment indicates that all agents are done

        Args:
            key (chex.PRNGKey): a random number
            state (EnvironmentState): the current environment state
            actions (Actions): the Actions as chosen by the agents

        Returns:
            Tuple[Observations, EnvironmentState, Rewards, Dict[str, bool], Infos]: a tuple with the new observations, environment state, rewards, dones and info after the step
        """

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        obs_re, states_re = self.reset(key_reset,epoch=state.epoch+1)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos



    def step_env(
        self, 
        key: chex.PRNGKey, 
        state: EnvironmentState, 
        actions: Actions,
    ) -> Tuple[Observations, EnvironmentState, Rewards, Dict[str, bool], Infos]:
        """ performs the actual implementation of the step function

        Args:
            key (chex.PRNGKey): a random number
            state (EnvironmentState): the current environment state
            actions (Actions): the Actions as chosen by the agents

        Returns:
            Tuple[Observations, EnvironmentState, Rewards, Dict[str, bool], Infos]: a tuple with the new observations, environment state, rewards, dones and info after the step
        """

        # Reward function
        def _calculate_reward(agent_id:str,cleared_market:doubleAuction.ClearedMarket,global_market:gloma.GlobalMarket)-> Reward:
            agent_index = self.agent_ids_to_index[agent_id]
            cleared_ask = cleared_market.cleared_asks[agent_index]
            cleared_bid = cleared_market.cleared_bids[agent_index]
            glob_market_ask = cleared_market.non_cleared_asks[agent_index]
            glob_market_bid = cleared_market.non_cleared_bids[agent_index]
            battery_ask = cleared_market.battery_shortage_asks[agent_index]
            battery_bid = cleared_market.battery_overflow_bids[agent_index]
            cte_feed_in_cost = jnp.where((glob_market_bid[ledger.OfferElemIndex.AMOUNT] + battery_bid[ledger.OfferElemIndex.AMOUNT]) > 0,
                                            global_market.feed_in_cte_cost,0)
            cte_time_of_use_cost = jnp.where((glob_market_ask[ledger.OfferElemIndex.AMOUNT] + battery_ask[ledger.OfferElemIndex.AMOUNT]) > 0,
                                             global_market.time_of_use_cte_cost,0)

            # agent gets rewarded on the money earned (positive) or the less money paid (negative)
            reward = (
                (cleared_bid[ledger.OfferElemIndex.PRICE] * cleared_bid[ledger.OfferElemIndex.AMOUNT]) - 
                (cleared_ask[ledger.OfferElemIndex.PRICE] * cleared_ask[ledger.OfferElemIndex.AMOUNT] )+ 
                (glob_market_bid[ledger.OfferElemIndex.PRICE] * glob_market_bid[ledger.OfferElemIndex.AMOUNT]) - 
                (glob_market_ask[ledger.OfferElemIndex.PRICE] * glob_market_ask[ledger.OfferElemIndex.AMOUNT] ) +
                (battery_bid[ledger.OfferElemIndex.PRICE] * battery_bid[ledger.OfferElemIndex.AMOUNT]) -
                (battery_ask[ledger.OfferElemIndex.PRICE] * battery_ask[ledger.OfferElemIndex.AMOUNT]) -
                cte_feed_in_cost - cte_time_of_use_cost # costs are always decreasing the agents reward
            )
            return reward

        
        # transform offers from actions to 2D jax array datatype required by DoubleAuction and globalMarket
        agent_indices = jnp.array([self.agent_ids_to_index[agent_id] for agent_id in self.agent_ids])
        offer_prices = jnp.ravel(jnp.array([actions[agent_id]["price"] for agent_id in self.agent_ids]))
        offer_amounts = jnp.ravel(jnp.array([actions[agent_id]["amount"] for agent_id in self.agent_ids]))
        unsorted_offers = jax.numpy.vstack([agent_indices, offer_prices, offer_amounts]).T
        offers = ledger.add_all_offers(unsorted_offers)
        
        # calculate the new battery levels and energy excess or shortage
        new_battery_values = _calculate_battery_level_and_excesses(state,self.agent_ids,offer_amounts)
        
        # clear market
        cleared_market = doubleAuction.clear_market(offers,self.num_agents)
        # clear remaining offers on Global Market
        cleared_market_with_global = gloma.clear_non_cleared_offers_on_global_market(self.global_market,state.timestep,cleared_market)
        # clear the battery excesses
        complete_cleared_market = gloma.add_battery_level_out_of_bounds(
                                                                    self.global_market,
                                                                    state.timestep,
                                                                    cleared_market_with_global,
                                                                    agent_indices,
                                                                    new_battery_values.shortage,
                                                                    new_battery_values.overflow)

        # calculate rewards
        rewards = {agent_id: _calculate_reward(agent_id,complete_cleared_market,self.global_market) for agent_id in self.agent_ids}
        
        # Collect all information used at the end of the training for analysis purposes
        # must contain a dictionary of 1D jax arrays with length equal to number of agents
        info = {
                    "epoch": jnp.full(self.num_agents,state.epoch),
                    "epoch_step":jnp.full(self.num_agents,state.timestep),
                    "market_cleared_price":jnp.full(self.num_agents,complete_cleared_market.price),
                    "cleared_price":complete_cleared_market.cleared_asks[:,ledger.OfferElemIndex.PRICE] + complete_cleared_market.cleared_bids[:,ledger.OfferElemIndex.PRICE], # each agent has or an asking or a bidding price
                    
                    "agent_raw_amount":jnp.array([actions[agent_id]["raw_amount"] for agent_id in self.agent_ids]),
                    "agent_ask_amount":jnp.array([jnp.where(actions[agent_id]["amount"] > 0, actions[agent_id]["amount"],0) for agent_id in self.agent_ids]),
                    "agent_bid_amount":jnp.array([jnp.where(actions[agent_id]["amount"] <= 0, - actions[agent_id]["amount"],0) for agent_id in self.agent_ids]),

                    "agent_raw_price":jnp.array([actions[agent_id]["raw_price"] for agent_id in self.agent_ids]),
                    "agent_price":jnp.array([actions[agent_id]["price"] for agent_id in self.agent_ids]),

                    "cleared_ask_amount":complete_cleared_market.cleared_asks[:,ledger.OfferElemIndex.AMOUNT],
                    "cleared_bid_amount":complete_cleared_market.cleared_bids[:,ledger.OfferElemIndex.AMOUNT],
                    "global_cleared_ask_amount":complete_cleared_market.non_cleared_asks[:,ledger.OfferElemIndex.AMOUNT],
                    "global_cleared_bid_amount":complete_cleared_market.non_cleared_bids[:,ledger.OfferElemIndex.AMOUNT],
                    "battery_shortage_ask_amount":complete_cleared_market.battery_shortage_asks[:,ledger.OfferElemIndex.AMOUNT],
                    "battery_overflow_bid_amount":complete_cleared_market.battery_overflow_bids[:,ledger.OfferElemIndex.AMOUNT],
                    "energy_demand": jnp.stack([state.energy_demand[a] for a in self.agent_ids]),
                    "battery_level": jnp.stack([state.battery_level[a] for a in self.agent_ids]), 
                    "max_energy_production_Wh": jnp.stack([state.max_energy_production_Wh[a] for a in self.agent_ids]),
                    "max_battery_capacity_Wh": jnp.stack([state.max_battery_capacity_Wh[a] for a in self.agent_ids]),
                    "time_of_use_price":jnp.full(self.num_agents,state.time_of_use_price),  
                    "feed_in_price":jnp.full(self.num_agents,state.feed_in_price),
                    "rewards":jnp.stack([rewards[a] for a in self.agent_ids]),  
                } 
                    
   
        #create new state
        new_timestep = state.timestep + 1
        glob_market_hour = self.global_market.hours[new_timestep]
        day_ahead_window_trigger = self.global_market.day_ahead_window_hour_trigger
        window = jnp.where(glob_market_hour == day_ahead_window_trigger,gloma.get_day_ahead_window(self.global_market,new_timestep),state.day_ahead_window)
        new_state = EnvironmentState(
                epoch = state.epoch,
                timestep = new_timestep,
                time_of_use_price = gloma.calc_variable_time_of_use_price(self.global_market,new_timestep),
                feed_in_price = gloma.calc_variable_feed_in_price(self.global_market,new_timestep),
                day_ahead_window=window,
                energy_demand  = {agent_id: self.agent_prosumers[agent_id].energy_consumption_Wh[new_timestep] for agent_id in self.agent_ids}, 
                max_energy_production_Wh = {agent_id:self.agent_prosumers[agent_id].max_energy_production_Wh for agent_id in self.agent_ids},
                battery_level = {agent_id:new_battery_values.battery_level[self.agent_ids_to_index[agent_id]] for agent_id in self.agent_ids},
                max_battery_capacity_Wh = {agent_id:self.agent_prosumers[agent_id].max_battery_capacity_Wh[new_timestep] for agent_id in self.agent_ids},
                max_battery_peak_Wh = {agent_id:self.agent_prosumers[agent_id].max_battery_peak_Wh[new_timestep] for agent_id in self.agent_ids} 
            ) 

        # create new observations based on new state
        new_obs = self.get_obs(new_state)
        dones = {a: state.timestep +1 >= self.epoch_nbr_of_steps for a in set(self.agent_ids).union({"__all__"})}
                
        return new_obs,new_state,rewards,dones,info
        
    
    




