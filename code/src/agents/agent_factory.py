""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the factory functions to create a specific agent object based on the given 
configuration dataclasses.
"""

from agents.agent_abstract import Agent
from agents.agent_config import AgentConfig, PpoAgentConfig, RuleBasedAgentConfig
from agents.ppo_agent import PpoAgent
from agents.ppo_selfplay_agent import SelfplayPpoAgent
from agents.rule_agent import RuleBasedAgent
from train_config import TrainConfig
from transformers.transformers_config import TransformerConfig


def create_agent(
        agent_config:AgentConfig,
        train_config:TrainConfig,
        transformer_config:TransformerConfig,
        action_space_size:int) -> Agent:
    """ Factory function to create an specific implementation of an agent object, based on the given configuration dataclasses

    Args:
        agent_config (AgentConfig): a specific AgentConfig dataclass, the type of child class of AgentConfig 
                                    will determine which agent implementation is returned
        train_config (TrainConfig): the TrainConfig dataclass containing all training specific parameters
        transformer_config (TransformerConfig): the TransformerConfig dataclass containing all transformer specific parameters
        action_space_size (int): the size of the action space

    Raises:
        Exception: raised when the AgentConfig child dataclass is not implemented in the factory function

    Returns:
        Agent: the Agent object instantiated from the Agent child class corresponding to the AgentConfig dataclass type 
    """
       
    if isinstance(agent_config,PpoAgentConfig):
        agent = PpoAgent(agent_config,train_config,transformer_config,action_space_size)
    
    elif isinstance(agent_config,RuleBasedAgentConfig):
        agent = RuleBasedAgent(agent_config,train_config,transformer_config,action_space_size)
    
    else:
        raise Exception("the agent configuration is not implemented")

    return agent



def create_selfplay_agent(
        agent_config:AgentConfig,
        train_config:TrainConfig,
        transformer_config:TransformerConfig,
        action_space_size:int,
        num_agents:int) -> Agent:
    """ Factory function to create an specific implementation of an selfplay agent object, based on 
        the given configuration dataclasses

    Args:
        agent_config (AgentConfig): a specific selfplay AgentConfig dataclass, the type of child class of AgentConfig 
                                    will determine which selfplay agent implementation is returned
        train_config (TrainConfig): the TrainConfig dataclass containing all training specific parameters
        transformer_config (TransformerConfig): the TransformerConfig dataclass containing all transformer specific parameters
        action_space_size (int): the size of the action space
        num_agents (int): the number of agents/actors that will share the selfplay agent

    Raises:
        Exception: raised when the selfplay AgentConfig child dataclass is not implemented in the factory function

    Returns:
        Agent: the selfplay Agent object instantiated from the selfplay Agent child class 
               corresponding to the selfplay AgentConfig dataclass type
    """

    if isinstance(agent_config,PpoAgentConfig):
        agent = SelfplayPpoAgent(agent_config,train_config,transformer_config,action_space_size,num_agents)
       
    else:
        raise Exception("the selfplay agent configuration is not implemented")

    return agent