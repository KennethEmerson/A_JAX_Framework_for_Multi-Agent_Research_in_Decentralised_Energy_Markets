""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains the functions used to plot the experimental results
The plot functions all get encapsulated into a 'generic_plot_decorator' which adds additional 
functionality to the plot e.g saving the plot scaling x and y axis 
"""

import json
import os
from typing import Callable, List, Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_SIZE = (10, 8)
BOX_ANCHOR = (0.5, -0.09)
LINETYPES = ["solid","dashed"]

# thesis colors based o the tableau-colorblind10 colo palette designed to facilitate 
# readers with color vision deficiency
CB_COLORS = {
    'dark_blue':'#006BA4', 
    'dark_orange':'#FF800E', 
    'grey':'#ABABAB', 
    'black':'#595959', 
    'blue':'#5F9ED1', 
    'red':'#C85200', 
    'dark_grey':'#898989', 
    'light_blue':'#A2C8EC', 
    'light_orange':'#FFBC79', 
    'light_grey':'#CFCFCF'}

COLOR_LIST = list(CB_COLORS.values())

#######################################################################################
# DATA LOADERS
#######################################################################################

def load_data(path:str) -> Tuple[dict,pd.DataFrame]:
    """ loads the logged data and configuration from the experiment

    Args:
        path (str): folder path where to find the config.json and "logging.csv"

    Returns:
        Tuple[dict,pd.DataFrame]: a tuple containing the config as a dict and the logging as a dataframe
    """
    with open(os.path.join(path,"config.json")) as f:
            config = json.load(f)
            f.close()
        
    with open(os.path.join(path,"logging.csv")) as f:
        df =pd.read_csv(f,header=0)
        f.close()
    return config, df



def load_run(experiment_path:str,postfix:str="_seed_0") -> pd.DataFrame:
    """loads the logged data and configuration from the experiment run with the given postfix.
       Be aware that if multiple sub folders exist in the experiment path with the same postfix, 
       the first sorted subfolder will be used.

    Args:
        experiment_path (str): path where the runs are stored
        postfix (str, optional): the postfix to search for in the sub folders. 
                                 This will be used to determine which run to load.
    Returns:
        pd.DataFrame: the dataframe containing the logged experiment data. 
    """
    subfolders = [ f.path for f in os.scandir(experiment_path) if f.is_dir() ]
    subfolders.sort()
    folder_seed_0  = list(filter(lambda x:x.endswith(postfix),subfolders))[0]
    _, experiment_df_seed_0 = load_data(folder_seed_0)
     
    return experiment_df_seed_0



def load_all_runs(experiment_path:str) -> List[pd.DataFrame]:
    """loads the logged data and configuration from one experiment that is all 5 runs
        with seeds 0,10,20,30,40,50. The folder must contain subfolders ending with
        the postfixes "_seed_0","_seed_10","_seed_20","_seed_30","_seed_40"
    
    Args:
        experiment_path (str): path where the runs for the experiment are stored
    
    Returns:
        pd.DataFrame: the list containing the dataframes with all logged experiment data. 
                      with one dataframe per run
    """
    subfolders = [ f.path for f in os.scandir(experiment_path) if f.is_dir() ]
    subfolders.sort()
    seeds_postfix = ["_seed_0","_seed_10","_seed_20","_seed_30","_seed_40"]

    folder_seed_0  = list(filter(lambda x:x.endswith(seeds_postfix[0]),subfolders))[0]
    folder_seed_10 = list(filter(lambda x:x.endswith(seeds_postfix[1]),subfolders))[0]
    folder_seed_20 = list(filter(lambda x:x.endswith(seeds_postfix[2]),subfolders))[0]
    folder_seed_30 = list(filter(lambda x:x.endswith(seeds_postfix[3]),subfolders))[0]
    folder_seed_40 = list(filter(lambda x:x.endswith(seeds_postfix[4]),subfolders))[0]

    _, experiment_df_seed_0 = load_data(folder_seed_0)
    _, experiment_df_seed_10 = load_data(folder_seed_10)
    _, experiment_df_seed_20 = load_data(folder_seed_20)
    _, experiment_df_seed_30 = load_data(folder_seed_30)
    _, experiment_df_seed_40 = load_data(folder_seed_40)

    return [experiment_df_seed_0,experiment_df_seed_10,experiment_df_seed_20,experiment_df_seed_30,experiment_df_seed_40]




#######################################################################################
# GENERIC PLOT DECORATOR
#######################################################################################

def generic_plot_decorator(plot_func:Callable) -> Callable:
    """decorator to encapsulate the individual partial plot functions, returns the encapsulated plot function
       The decorated function should work solely on the pyplot axis object, provided by the decorator function 
       or by the calling function providing the tha axis object
    Args:
        plot_func (Callable): the partial plot function that manipulates the pyplot axis object

    Returns:
        Callable: the encapsulated plot function
    """
      
    def inner_plot(
        df:pd.DataFrame,
        sample_min:int=0,
        sample_max:Optional[int]=None,
        y_limit:Optional[Tuple[float,float]]=None,
        save:bool=False,
        experiment_path:str="",
        experiment:str="",
        fileformat:str="pdf",
        ax:Optional[matplotlib.axes.Axes]=None,
        **kwargs:dict
    ) -> None:
        """inner decorator function for the experiment plots, please check the inner function to determine
        the behaviour of the function. The decorator adds additional generic functionality to the plot function
        such as limiting the x and y axis limits and saving the plot

        Args:
            df (pd.DataFrame): the dataframe containing the experiment logged metrics
            sample_min (int, optional): min sample on x-axis. Defaults to 0.
            sample_max (Optional[int], optional): max sample on x-axis. Defaults to None.
            y_limit (Optional[Tuple[float,float]], optional): tuple containing y-axis min and max. Defaults to None.
            save (bool, optional): if true, plot is saved in experiment_path. Defaults to False.
            experiment_path (str, optional): folder path in which to store the plot. Defaults to "".
            experiment (str, optional): experiment name used to determine filename in which to store the plot. Defaults to "".
            fileformat (str): the file format e.g. PDF,png,... to be used by matplotlib to store the plot. defaults to PDF
            ax (matplotlib.axes.Axes, optional): if given the underlying decorated plot will be created within the existing plt axis
                                                 if not provided a new standalone plot will be created. This enables using the 
                                                 underlying decorated plot to be used as a sub figure. Defaults to None
            kwargs (dict): additional arguments to be sent to underlying decorated plot function
        Raises:
            Exception: raised when not all required saving parameters are provided
            """
        
        # check parameters to store plot
        if save and (not experiment or not experiment_path):
            raise Exception("to save the plot both parameters experiment and experiment_path must be provided")
   
        
        # filter out all samples not included between sample min and sample max
        sample_window = df["sample"].max()
        if sample_max:
            sample_window = sample_max
        temp_df = df[(df["sample"]>=sample_min) & (df["sample"]<sample_window)]
        
        # create a new fig and axis if none was provided, this will result in a standalone plot
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            show = True
        
        # limit y axis if parameter was passed
        if y_limit:
            ax.set_ylim(*y_limit) # type: ignore
                        
        # execute the decorated function
        title = plot_func(temp_df,ax,kwargs)

        # create the filename
        filename = f"{experiment + '_'if experiment else ''}"
        filename = filename + title if title else ''
        if sample_min or sample_max:
            s_min = sample_min if sample_min else "0"
            s_max = sample_max if sample_max else "Na" 
            filename = filename + f"_x_limit_{s_min}_{s_max}"
        if y_limit: 
            filename = filename + f"_y_limit_{y_limit[0]}_{y_limit[1]}" 
        filename = filename + f".{fileformat}"
               
        if save:
            plt.savefig(os.path.join(experiment_path,filename),format=fileformat)
        if show:
            plt.show()
    return inner_plot
   


#######################################################################################
# PARTIAL PLOTS AND THEIR DECORATED FUNCTIONS used in overview
#######################################################################################

def partial_plot_mean_epoch_returns_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={}) -> str:
    """undecorated function to plot the total mean return per epoch per agent

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axes): the matplotlib axis to plot the graph on
    """             
    sample_indices = temp_df.groupby(["epoch"])['sample'].first().reset_index()
    grouped_df_agents= temp_df.groupby(["agent_index","epoch"])["rewards"].mean().reset_index().rename(columns={"rewards":"rewards_mean"})

    grouped_df= temp_df.groupby(["epoch"])["rewards"].mean().reset_index().rename(columns={"rewards":"rewards_mean"})
    
    for i in grouped_df_agents["agent_index"].unique():
        x = sample_indices["sample"]
        y = grouped_df_agents[grouped_df_agents["agent_index"]==i]["rewards_mean"]
        j = int(i)%2 
        ax.step(
            x,y,
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")

    ax.step(sample_indices["sample"],grouped_df["rewards_mean"],linewidth=2,color="black",label="all agents")
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean reward per epoch [€]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean reward per epoch")

    return "mean_epoch_returns_per_agent"
plot_mean_epoch_returns_per_agent = generic_plot_decorator(partial_plot_mean_epoch_returns_per_agent) 

def partial_plot_mean_energy_amount_offered_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={})-> str:
    """undecorated function to plot the mean of energy offers per epoch per agent

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axe): the matplotlib axis to plot the graph on
    """
    df = temp_df.copy()              
    samples = df.groupby(["epoch"])['sample'].first().reset_index()
    df["offers"] = df["agent_ask"] -df["agent_bid"]
    mean = df.groupby(["epoch","agent_index"])["offers"].mean().reset_index().rename(columns={"offers":"offers_mean"})
    
    for i  in temp_df["agent_index"].unique():
        j = int(i)%2 
        ax.step(
            samples["sample"],
            mean[mean["agent_index"]==i]["offers_mean"],
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")
       
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean amount [Wh]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean offered amount per epoch")

    return "mean_energy_amount_offered_per_agent"
plot_mean_energy_amount_offered_per_agent = generic_plot_decorator(partial_plot_mean_energy_amount_offered_per_agent) 

def partial_plot_mean_amount_internally_cleared_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={})-> str:
    """undecorated function to plot mean amount cleared on internal market per epoch

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axe): the matplotlib axis to plot the graph on
    """
    
    df = temp_df.copy()    
    samples = df.groupby(["epoch"])['sample'].first().reset_index()
    df["internal_cleared"] = df["cleared_ask"] -df["cleared_bid"]
    mean = df.groupby(["epoch","agent_index"])["internal_cleared"].mean().reset_index().rename(columns={"internal_cleared":"internal_cleared_mean"})
    
    for i  in temp_df["agent_index"].unique():
        j = int(i)%2 
        ax.step(
            samples["sample"],
            mean[mean["agent_index"]==i]["internal_cleared_mean"],
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")
       
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean amount [Wh]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean amount cleared on internal market per epoch")

    return "mean_amount_internally_cleared_per_agent"
plot_mean_amount_internally_cleared_per_agent = generic_plot_decorator(partial_plot_mean_amount_internally_cleared_per_agent) 

def partial_plot_mean_amount_global_cleared_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={})-> str:
    """undecorated function to plot mean amount cleared on global market per epoch

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axe): the matplotlib axis to plot the graph on
    """
    
    df = temp_df.copy()    
    samples = df.groupby(["epoch"])['sample'].first().reset_index()
    df["global_cleared"] = df["global_cleared_ask"] -df["global_cleared_bid"]
    mean = df.groupby(["epoch","agent_index"])["global_cleared"].mean().reset_index().rename(columns={"global_cleared":"global_cleared_mean"})
    
    color = 0
    for i  in temp_df["agent_index"].unique():
        j = int(i)%2 
        ax.step(
            samples["sample"],
            mean[mean["agent_index"]==i]["global_cleared_mean"],
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")
        color +=1
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean amount [Wh]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean amount cleared on global market per epoch")

    return "mean_amount_global_cleared_per_agent"
plot_mean_amount_global_cleared_per_agent = generic_plot_decorator(partial_plot_mean_amount_global_cleared_per_agent) 

def partial_plot_mean_batt_discrepancy_amount_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={})-> str:
    """undecorated function to plot the mean of energy offers per epoch per agent

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axe): the matplotlib axis to plot the graph on
    """
    
    df = temp_df.copy()    
    samples = df.groupby(["epoch"])['sample'].first().reset_index()
    df["discrepancy"] = df["battery_shortage_ask"] -df["battery_overflow_bid"]
    mean = df.groupby(["epoch","agent_index"])["discrepancy"].mean().reset_index().rename(columns={"discrepancy":"discrepancy_mean"})
    
    color = 0
    for i  in temp_df["agent_index"].unique():
        j = int(i)%2 
        ax.step(
            samples["sample"],
            mean[mean["agent_index"]==i]["discrepancy_mean"],
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")
        color +=1
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean amount [Wh]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean battery shortage/overflow per epoch")

    return "mean_batt_discrepancy_amount_per_agent"
plot_mean_batt_discrepancy_amount_per_agent = generic_plot_decorator(partial_plot_mean_batt_discrepancy_amount_per_agent) 

def partial_plot_mean_battery_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={})-> str:
    """undecorated function to plot mean battery level per epoch

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axe): the matplotlib axis to plot the graph on
    """
    
    df = temp_df.copy()    
    samples = df.groupby(["epoch"])['sample'].first().reset_index()
    mean = df.groupby(["epoch","agent_index"])["battery_level"].mean().reset_index().rename(columns={"battery_level":"battery_level_mean"})
     
    
    color = 0
    for i  in temp_df["agent_index"].unique():
        j = int(i)%2
       
        ax.step(
            samples["sample"],
            mean[mean["agent_index"]==i]["battery_level_mean"],
            linewidth=2,alpha= 0.7,linestyle = LINETYPES[j],color=COLOR_LIST[int(i)],label=f"agent {int(i):02d}")
        color +=1
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean amount [Wh]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("mean battery level per epoch")

    return "mean_battery_per_agent"

plot_mean_battery_per_agent = generic_plot_decorator(partial_plot_mean_battery_per_agent)

#######################################################################################
# prices scatter plots
#######################################################################################

def partial_scatter_plot_ask_bids_prices(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={}) -> str:
    """undecorated function to plot the offered prices divided by asks and bids vs the global market price 

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axes): the matplotlib axis to plot the graph on
    """        
    agent_ask_indices = temp_df[temp_df["agent_ask"]>0]["sample"]
    agent_ask_prices = temp_df[temp_df["agent_ask"]>0]["agent_price"]
    agent_bid_indices = temp_df[temp_df["agent_bid"]>0]["sample"]
    agent_bid_prices = temp_df[temp_df["agent_bid"]>0]["agent_price"]
    
    samples = temp_df[temp_df["agent_index"]==0]["sample"]
    feed_in_prices = temp_df[temp_df["agent_index"]==0]["feed_in_price"]
    time_of_use_prices = temp_df[temp_df["agent_index"]==0]["time_of_use_price"]
           
    ax.scatter(
        agent_ask_indices,
        agent_ask_prices,
        s=0.5,
        color=CB_COLORS["dark_blue"],
        label="agent ask prices"
        )
    ax.scatter(
        agent_bid_indices,
        agent_bid_prices,
        s=0.5,
        color=CB_COLORS["dark_orange"],
        alpha=0.5,label="agent bid prices")
    ax.fill_between(
        samples,
        feed_in_prices,
        time_of_use_prices,
        facecolor=CB_COLORS["grey"],
        edgecolor=None,
        alpha=0.5,
        label="global market")
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("price [€]")
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("ask and bid prices offered by agents")
   
    return "ask_bids_prices"
scatter_plot_ask_bids_prices = generic_plot_decorator(partial_scatter_plot_ask_bids_prices) 

def partial_scatter_plot_agent_prices(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={}) -> str:
    """undecorated function to plot the offered prices divided by asks and bids vs the global market price 

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axes): the matplotlib axis to plot the graph on
    """        
        
    samples = temp_df[temp_df["agent_index"]==0]["sample"]
    feed_in_prices = temp_df[temp_df["agent_index"]==0]["feed_in_price"]
    time_of_use_prices = temp_df[temp_df["agent_index"]==0]["time_of_use_price"]

    for i in temp_df["agent_index"].unique():    
        ax.scatter(
            samples,
            temp_df[temp_df["agent_index"]==i]["agent_price"], 
            s=0.5,
            alpha=0.7,
            color=COLOR_LIST[int(i)],
            label=f"agent {int(i):02d}"
            )
   
    ax.fill_between(
        samples,
        feed_in_prices,
        time_of_use_prices,
        facecolor=CB_COLORS["grey"],
        edgecolor=None,
        alpha=0.5,
        label="global market")
    
    ax.set_xlabel("timestep")
    ax.set_ylabel("price [€]")
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("prices offered by agents")
   
    return "offer_prices_per_agent"
scatter_plot_agent_prices = generic_plot_decorator(partial_scatter_plot_agent_prices) 

def partial_scatter_plot_cleared_market_prices(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,_:dict={}) -> str:
    """undecorated function to plot the cleared internal market prices

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axes): the matplotlib axis to plot the graph on
    """         
    filter_indices = (temp_df["agent_index"]==0) & ((temp_df["market_cleared_price"] !=0))
    cleared_price_indices = temp_df[filter_indices]["sample"]
    cleared_prices = temp_df[filter_indices]["market_cleared_price"]
        
    glob_price_indices = temp_df[(temp_df["agent_index"]==0)]["sample"]
    feed_in_prices = temp_df[(temp_df["agent_index"]==0)]["feed_in_price"]
    time_of_use_prices = temp_df[(temp_df["agent_index"]==0)]["time_of_use_price"]
            
    ax.scatter(cleared_price_indices,cleared_prices,s=1,color=CB_COLORS["dark_blue"],label="cleared IM prices")
    ax.fill_between(glob_price_indices,feed_in_prices,time_of_use_prices,facecolor=CB_COLORS["grey"],edgecolor=None,alpha=0.5,label="global market")
    ax.set_xlabel("timestep")
    ax.set_ylabel("price [€]")
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=BOX_ANCHOR)
    ax.title.set_text("price of the internal market clearing")

    return "cleared_market_prices"
scatter_plot_cleared_market_prices = generic_plot_decorator(partial_scatter_plot_cleared_market_prices) 


def partial_plot_offer_price_per_agent(temp_df:pd.DataFrame,ax:matplotlib.axes.Axes,kwargs:dict) -> str:
    """undecorated function to plot the offer price per agent

    Args:
        temp_df (pd.DataFrame): the dataframe containing the experiment logged metrics
        ax (matplotlib.axes.Axes): the matplotlib axis to plot the graph on
        kwargs (dict): additional kwargs parameter containing the "agent_index" key
    """            
    agents_index = kwargs["agent_index"]
    
    y = temp_df[(temp_df["agent_index"] == agents_index)]["agent_price"]
    x = temp_df[(temp_df["agent_index"] == agents_index)]["sample"]
    feed_in_prices = temp_df[(temp_df["agent_index"]==agents_index)]["feed_in_price"]
    time_of_use_prices = temp_df[(temp_df["agent_index"]==agents_index)]["time_of_use_price"]
    
    ax.scatter(x,y,linewidths=1,color=CB_COLORS["dark_blue"],s=2,alpha=0.9,label=f"offered prices for agent {int(agents_index):02d}" )
    ax.fill_between(x,feed_in_prices,time_of_use_prices,facecolor=CB_COLORS["grey"],edgecolor=None,alpha=0.5,label="global market")
    ax.set_xlabel("timestep")
    ax.set_ylabel('price [€]')
    ax.title.set_text(f"offered prices for agent {int(agents_index):02d}")

    return "offer_price_per_agent"

plot_offer_price_per_agent = generic_plot_decorator(partial_plot_offer_price_per_agent) 


#######################################################################################
# COMPLETE PLOTS AS USED IN THESIS
#######################################################################################

def plot_mean_returns_per_epoch_all_seeds(
        data:List[pd.DataFrame],
        experiment:str,
        save:bool=False,
        plot_path:str="",
        fileformat:str="pdf",
        figsize:Tuple[int,int]=(15,8)
        ) -> None:
    """ plots the mean rewards per epoch for all experiment runs

    Args:
        data (List[pd.DataFrame]): the list with loaded logging data one dataframe for each run
        experiment (str): the name of the experiment
        save (bool, optional): if true the plot gets saved in the plot_path. Defaults to False.
        plot_path (str, optional): the folder path where to store the files if save is selected. Defaults to "".
        fileformat (str, optional): _description_. Defaults to "pdf".
        figsize (Tuple[int,int]): size of plot. defaults to (15,8)
    """
    
    if save and (not experiment or not plot_path):
            raise Exception("to save the plot both parameters experiment and experiment_path must be provided")
    
    color_list = list(CB_COLORS.values())
       
    fig, ax = plt.subplots(figsize=figsize)
   
    color = 0 
    for df in data:
        samples = df.groupby(["epoch"])['sample'].first().reset_index()
        grouped_df= df.groupby(["epoch"])["rewards"].mean().reset_index().rename(columns={"rewards":"rewards_mean"})
                
        ax.step(
            samples["sample"],
            grouped_df["rewards_mean"],
            linewidth=2,alpha= 0.7,color=color_list[color], # type:ignore
            label=f"mean overall reward: seed {color*10}") 
        color += 1

    ax.set_xlabel("timestep")
    ax.set_ylabel("mean reward per epoch [€]")
    ax.set_ylim([None, 0.005])
    ax.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, -0.07),)
    
    #plt.subplots_adjust(right=0.70,bottom=0.15)
    fig.text(0.90, -0.05, experiment, ha='right', fontsize=7)
    if save:
        filename = f"{experiment}_mean_rewards_all_seeds.{fileformat}"
        plt.savefig(os.path.join(plot_path,filename), format=fileformat,bbox_inches='tight') 
    
    plt.show()


def plot_last_epoch_boxplot(
        df:pd.DataFrame,
        experiment:str,
        save:bool=False,
        plot_path:str="",
        fileformat:str="pdf",
        figsize:Tuple[int,int]=(15,8)
        )-> None:
    """ plots a boxplot for the rewards of the last epoch

    Args:
        df (pd.DataFrame): logging data 
        experiment (str): the name of the experiment
        save (bool, optional): if true the plot gets saved in the plot_path. Defaults to False.
        plot_path (str, optional): the folder path where to store the files if save is selected. Defaults to "".
        fileformat (str, optional): _description_. Defaults to "pdf".
        figsize (Tuple[int,int]): size of plot. defaults to (15,8)

    """
    if save and (not experiment or not plot_path):
            raise Exception("to save the plot both parameters experiment and experiment_path must be provided")
    
    last_epoch = df["epoch"].max()
    labels = ["all agents", "agent 00","agent 01","agent 02","agent 03","agent 04","agent 05","agent 06","agent 07"]
    rewards = [
        df[df["epoch"]==last_epoch]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==0)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==1)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==2)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==3)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==4)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==5)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==6)]["rewards"],
        df[(df["epoch"]==last_epoch)&(df["agent_index"]==7)]["rewards"],
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bplot = ax.boxplot(
        rewards,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        labels = labels  # fill with color
                   ) 
    for i, median in enumerate([np.median(reward) for reward in rewards]):
        ax.text(i + 1, median + 0.0002, f'{median:.5f}', ha='center', va='bottom', fontsize= 8)
    for patch in bplot['boxes']:
         patch.set_facecolor(CB_COLORS["light_grey"])

    fig.text(0.9, 0.05, experiment, ha='right', fontsize=figsize[0]*2/3)
    ax.set_ylabel('reward [€]')    
    if save:
        filename = f"{experiment}_boxplot.{fileformat}"
        plt.savefig(os.path.join(plot_path,filename), format=fileformat,bbox_inches='tight') 
    
    plt.show()