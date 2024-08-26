""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains the actual training loop 

Based on JaxMARL Implementation of PPO: 
https://github.com/FLAIROx/JaxMARL/blob/main/baselines/IPPO/ippo_ff_overcooked.py
"""

from typing import Any, Dict, Sequence, Tuple, Union

import jax
import train_helpers as train_h
from agents.agent_abstract import Agent
from agents.agent_config import AgentConfig
from agents.agent_factory import create_agent, create_selfplay_agent
from environments.energymarket_env import DoubleAuctionEnv
from experiments.experiment_config import ExperimentConfig
from flax.training.train_state import TrainState
from train_config import TrainConfig
from train_data_objects import (
    _RunnerState,
    _Transition,
    _TransitionAllAgents,
)
from transformers.transformers_config import TransformerConfig


def train(
        train_config:TrainConfig,
        transformer_config:Union[TransformerConfig,Dict[str,TransformerConfig]],
        env:DoubleAuctionEnv,
        agent_configs:Union[AgentConfig,Sequence[AgentConfig]],
        rng:jax.Array) -> Dict:
    """ the training loop orchestrating all interactions between agents and environment and the actual learning of the agents
        the function consists of multiple if then else branches providing two different processing flows depending of selfplay or multi agent mode
        NOTE: the main training loop is written in plain Python and does not get compiled using the JAX JIT compiler.
            all functions called, related to the agents, Prosumers, Environment, Transformers are however JIT compatible and do get compiled  

    Args:
        train_config (TrainConfig): the dataclass containing all training related parameters
        transformer_config (Union[TransformerConfig,Dict[str,TransformerConfig]]): the dataclass containing all Transformer related parameters when using selfplay 
                                                                                   or a dict mapping agent identifiers to their TransformerConfig dataclasses 
        env (DoubleAuctionEnv): the Environment object
        agent_configs (Union[AgentConfig,Sequence[AgentConfig]]): dataclasses containing the Agent parameters for selfplay or
                                                                  list of Agentconfig dataclasses containing the Agent parameters, one for each agent
        rng (jax.Array): a random number

    Raises:
        Exception: raised when selfplay parameters are ill configured
        
    Returns:
        Dict: a dict of form {"runner_state": runner_state, "metrics": metric}
    """
    

    ###################################################
    # INITIALISATION
    ###################################################

    if train_config.SELF_PLAY and isinstance(agent_configs,AgentConfig):
        is_selfplay = True
    elif not train_config.SELF_PLAY and  isinstance(agent_configs,list) and len(agent_configs) == env.num_agents:
        is_selfplay = False   
        agent_config_list = agent_configs
    else:
        raise Exception("selfplay configuration error, check selfplay setting and number of agents in the config.AGENTS")

    num_actors = env.num_agents
    agent_id_list = env.agent_ids 

    # initialise selfplay agent, the transformers and the trainstate
    if is_selfplay and isinstance(transformer_config,TransformerConfig) and isinstance(agent_configs,AgentConfig):                       
        transformers = train_h.create_selfplay_transformers(transformer_config,env)
        selfplay_agent = create_selfplay_agent(agent_configs,train_config,transformer_config,transformers[list(transformers.keys())[0]].action_space_size,env.num_agents)      
        rng, _rng = jax.random.split(rng)     
        selfplay_agent_train_state = selfplay_agent.create_init_train_state(transformers[list(transformers.keys())[0]].observation_space_size,_rng)
    

    # initialise multi agent agents, their transformers and trainstates
    elif not is_selfplay and isinstance(transformer_config,dict) and isinstance(agent_config_list,list):  
        multi_agents:Dict[str,Agent] ={}
        multiagent_train_states = {}
        rng, _rng = jax.random.split(rng)

        transformers = train_h.create_multiagent_transformers(transformer_config,env) 

        for mulitagent_config in agent_config_list:
            agent_id = mulitagent_config.ID
            
            agent = create_agent(mulitagent_config,train_config,transformer_config[agent_id],transformers[agent_id].action_space_size)
            rng, _rng = jax.random.split(rng)      
            train_state = agent.create_init_train_state(transformers[agent_id].observation_space_size,_rng)
            
            multi_agents[agent_id] = agent
            multiagent_train_states[agent_id] = train_state
    

    # initialise the environment 
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng)
    


    ###################################################
    # ENVIRONMENT STEP IMPLEMENTATION 
    ###################################################  
  
    def _env_step(
            runner_state:_RunnerState,
            unused:Any) -> Tuple[_RunnerState,Union[_Transition,_TransitionAllAgents]]:
        """ internal function executing one environment step 
        """    
        train_states, env_state, last_obs, rng = runner_state
        last_transformed_obs = train_h.transform_observations(
                                                    agent_id_list=agent_id_list,
                                                    transformers=transformers,
                                                    state=env_state,
                                                    observations=last_obs,
                                                    )
        
        # select actions
        if is_selfplay and isinstance(train_states,TrainState):
            last_transformed_obs_batch = train_h.batchify(last_transformed_obs, env.agent_ids, num_actors)
            rng, _rng = jax.random.split(rng)
            pi, selfplay_value, selfplay_action, selfplay_log_prob = selfplay_agent.select_action(train_states,last_transformed_obs_batch,_rng)               
            agents_actions = train_h.unbatchify(selfplay_action, env.agent_ids, env.num_agents)
            agents_actions = {k:v.flatten() for k,v in agents_actions.items()}
            
        
        elif not is_selfplay and isinstance(train_states,Dict):
            agents_actions = {}
            multiagent_values = {}
            multiagent_log_probs = {}

            for agent_id in multi_agents.keys():
                agent_obs = last_transformed_obs[agent_id]
                
                agent = multi_agents[agent_id]
                train_state = train_states[agent_id]
                        
                # select action
                rng, _rng = jax.random.split(rng)
                pi, multiagent_value, multiagent_action, multiagent_log_prob = agent.select_action(train_state,agent_obs,_rng)               
            
                agents_actions[agent_id] = multiagent_action
                multiagent_values[agent_id] = multiagent_value
                multiagent_log_probs[agent_id] = multiagent_log_prob
    
        else:
            raise Exception("the trainstates object should be a Trainstate when using selfplay or a dict of trainstates otherwise")
        
        # transform actions
        env_act = train_h.transform_actions(
                        agent_id_list= agent_id_list,
                        transformers=transformers,
                        state=env_state,
                        actions=agents_actions,
                        )

        # infer actions on environment
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, env_act)
                        
        # get all values
        info = jax.tree_map(lambda x: x.reshape((num_actors)), info)
        if is_selfplay:
            selfplay_transition = _Transition(
                        train_h.batchify(done, env.agent_ids, num_actors).squeeze(),
                        selfplay_action,
                        selfplay_value,
                        train_h.batchify(reward, env.agent_ids, num_actors).squeeze(),
                        selfplay_log_prob,
                        last_transformed_obs_batch,
                        info   
                    )
            runner_state = _RunnerState(train_states, env_state, obsv, rng)
            
            # the state will be used as state in the jax.lax.scan
            # the transitions will be stacked in the jax.lax.scan
            return runner_state, selfplay_transition
        
        else:
            transition = _TransitionAllAgents(
                            done,
                            agents_actions,
                            multiagent_values,
                            reward,
                            multiagent_log_probs,
                            last_transformed_obs,
                            info   
                        )
        
            runner_state = _RunnerState(train_states, env_state, obsv, rng)
            
            # the state will be used as state in the jax.lax.scan
            # the transitions will be stacked in the jax.lax.scan
            return runner_state, transition



    ###################################################
    # TRAINING ITERATION IMPLEMENTATION 
    ###################################################

    def _update_step(runner_state:_RunnerState, unused:Any) -> Tuple[_RunnerState,jax.Array]:
        """ internal function executing one full training iteration
        """    
        # run all actions in runner_state sequentially in environment
        runner_state, traj_batch_all_agents = jax.lax.scan(_env_step, runner_state, None, train_config.NUM_STEPS)


        # transform all observations received from the batch run of env steps
        train_states, env_state, last_obs, rng = runner_state
        last_obs_batch = train_h.transform_observations(
                            agent_id_list= agent_id_list,
                            transformers=transformers,
                            state=env_state,
                            observations=last_obs,
                            )
        

        # update the agent parameters based on their specific implementation
        rng, _rng = jax.random.split(rng)
        if is_selfplay and isinstance(train_states,TrainState) and isinstance(traj_batch_all_agents,_Transition) :
            last_obs_batch = train_h.batchify(last_obs_batch,env.agent_ids,num_actors)    
            update_state, loss_info = selfplay_agent.update_networks_for_one_epoch(train_states,last_obs_batch,traj_batch_all_agents,_rng)
            train_state = update_state[0]
            traj_batch = traj_batch_all_agents
            new_train_states = update_state[0]

        elif not is_selfplay and isinstance(train_states,dict) and isinstance(traj_batch_all_agents,_TransitionAllAgents) :
            new_train_states = {}

            for agent_id in multi_agents.keys():
                last_obs_batch_per_agent= last_obs_batch[agent_id]
                train_state = train_states[agent_id]
                # NOTE: shape of each transition element contains the values for one agent and all NUM_STEPS
                traj_batch = _Transition(
                    traj_batch_all_agents.done[agent_id],
                    traj_batch_all_agents.action[agent_id],
                    traj_batch_all_agents.value[agent_id],
                    traj_batch_all_agents.reward[agent_id],
                    traj_batch_all_agents.log_prob[agent_id],
                    traj_batch_all_agents.obs[agent_id],
                    traj_batch_all_agents.info
                )       
                update_state, loss_info = multi_agents[agent_id].update_networks_for_one_epoch(train_state,last_obs_batch_per_agent,traj_batch,_rng)
                new_train_states[agent_id] = update_state[0]

        else:
            raise Exception("the trainstates object should be a Trainstate when using selfplay or a dict of trainstates otherwise")


        metric = traj_batch.info
        rng = update_state[-1]
        runner_state = _RunnerState(new_train_states, env_state, last_obs, rng)
        return runner_state, metric



    ###################################################
    # MAIN TRAINING LOOP IMPLEMENTATION 
    ###################################################

    _, _rng = jax.random.split(rng)
    if is_selfplay:
        runner_state = _RunnerState(selfplay_agent_train_state, env_state, obsv, _rng)
    else:
        runner_state = _RunnerState(multiagent_train_states, env_state, obsv, _rng)
    
    # MAIN TRAINING LOOP
    runner_state, metric = jax.lax.scan(
        _update_step, runner_state, None, train_config.TOTAL_TIMESTEPS // train_config.NUM_STEPS
    )
    return {"runner_state": runner_state, "metrics": metric}


