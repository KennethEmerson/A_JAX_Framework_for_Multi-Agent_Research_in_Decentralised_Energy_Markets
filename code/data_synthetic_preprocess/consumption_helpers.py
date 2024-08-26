""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
File contains all helper functions to create the household consumption profiles and data
    
"""
import configparser
import os
import random
import subprocess
from os import listdir
from os.path import isfile
from typing import Dict, List, Optional

import pandas as pd

###################################################################
# retrieve contants for config.ini
###################################################################
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))

ACTOR_NAME_PREFIX = config['GENERAL']['actor_prefix']
OUTPUT_PATH = config['SYNTHETIC_DATA']['synthetic_data_folder']
AGENTS_DIRECTORY = config['SYNTHETIC_DATA']['consumption_actors_subfolder']
SECONDS_DIRECTORY = config['SYNTHETIC_DATA']['consumption_data_seconds_subfolder']
HOURS_DIRECTORY = config['SYNTHETIC_DATA']['consumption_data_hours_subfolder']
antgen_path = os.path.join(os.getcwd(),"data_synthetic_preprocess","antgen")


###################################################################
# Helper functions
###################################################################
def create_random_houses(nbr_of_houses:int,
                         start_nbr_house:int=0,
                         users_path:str="users",
                         max_residents:int=6) -> List[str]:
    """ creates a random household configuration file containing a list of household member 
        profiles and a baseload profile. It uses predefined user and baseload profiles which must be
        stored in a specific folder (default "users") and must conform to the schema required by the ANTgen library 
        (For more info on ANTgen, please consult: https://gitlab.com/nunovelosa/antgen)

    Args:
        nbr_of_houses (int): nbr of households to create
        start_nbr_house (int, optional): the first index to use for the first profile filename. Defaults to 0.
        users_path (str, optional): the directory where the user profiles to be used, are stored. Defaults to "users".
        max_residents (int, optional): the max numbers of residents in any given household profile. Defaults to 6.

    Returns:
        List[str]: a list of actor names for which a file is created
    """

    # get all the user profiles from the provided directory
    onlyfiles = [f for f in listdir(os.path.join(os.path.dirname(__file__),users_path)) 
                 if isfile(os.path.join(os.path.dirname(__file__),users_path, f))]

    # extract the baseload and user profile files
    baseloads = list(filter(lambda k: 'baseload' in k, onlyfiles))
    users = list(filter(lambda k: 'baseload' not in k, onlyfiles))

    # create the destination folder based on the configuration in config.ini
    actor_path = os.path.join(os.getcwd(),OUTPUT_PATH,AGENTS_DIRECTORY)
    if not os.path.exists(actor_path):
        os.makedirs(actor_path)

    # create the actual profiles
    actor_list = []
    for i in range(start_nbr_house, start_nbr_house + nbr_of_houses): 
        
        # randomly choose nbr of users, which user profiles and a baseload
        nbr_of_residents = random.randint(1, max_residents)  # noqa: S311
        rand_users = random.sample(users, nbr_of_residents)
        rand_baseload = random.choice(baseloads)  # noqa: S311

        # create the configuration
        user_dict={f'resident{count}':value for count, value in enumerate(rand_users)}
        user_dict['baseload']= rand_baseload
        config_creator = configparser.ConfigParser()
        config_creator['GENERAL'] = {'name': "Sample configuration file"}
        config_creator['users'] = user_dict
            
        # Write the configuration to a file
        actor = f'{ACTOR_NAME_PREFIX}_{i:02d}'
        actor_list.append(actor)
        with open(os.path.join(actor_path  ,f'{actor}.conf'), 'w') as configfile:
            config_creator.write(configfile)
 
    return actor_list



def simulate_energy_usage_per_house(nbr_of_days:int,
                                    startdate:str,
                                    actor_list:Optional[List[str]]=None,
                                    baseload:str = "C10") -> None:
    """ Uses the ANTgen library to generate synthetic consumption data (in Wh) on a per second basis
        and stores the result in a set of files one for each user, appliance and activity and a file with the total consumption.
        We will only use the total consumption file to create the actual data.
        (For more info on ANTgen, please consult: https://gitlab.com/nunovelosa/antgen)

    Args:
        nbr_of_days (int): the number of days that get generated in one iteration. 
                            is used to limit the memory usage of the ANTgen library.
        startdate (str): the startdate for which to create the synthetic data
        actor_list (Optional[List[str]], optional): the list of actor household configuration files in the 
                                                    actors directory (as defined in config.ini) for which data must be created. 
                                                    If no list is provided, data will be created for all configuration files 
                                                    in the folder. Defaults to None.
        baseload (str, optional): the noise argument used by ANTgen, adds a baseload or noise to the energy consumption.
                                  must be one letter (C=cte, G=Gaussian) followed by the amplitude in Watt. Defaults to "C10".
    """
    
    # if no actor list is provided, create one based on all files found in the actors directory (as defined in config.ini)
    if not actor_list:
        actor_list = [
            name for name in os.listdir(os.path.join(OUTPUT_PATH,AGENTS_DIRECTORY)) 
            if os.path.isfile(os.path.join(OUTPUT_PATH,AGENTS_DIRECTORY, name)) and name.startswith(ACTOR_NAME_PREFIX) and name.endswith(".conf")]
        actor_list = list(map(lambda x: os.path.splitext(x)[0],actor_list))
    
    
    # call the ANTgen library to create and store the data
    for actor in actor_list:
        print(f"simulating {actor} for {nbr_of_days} days from {startdate}")
        subprocess.run([  # noqa: S603, S607
            "python3", f"{antgen_path}/main.py",
            "-n", baseload, 
            "-d", str(nbr_of_days), 
            "-o", os.path.join(OUTPUT_PATH,SECONDS_DIRECTORY,actor),
            "-w",
            "-b", startdate,
            "-m", os.path.join(antgen_path,"mapping.conf"),
            os.path.join(OUTPUT_PATH,AGENTS_DIRECTORY,f"{actor}.conf")],stderr=subprocess.DEVNULL)
              


def convert_energy_usage_per_house_to_hours(actor_list:Optional[List[str]]=None,
                                            append:bool=False,
                                            only_total:bool=True) -> None:
    """takes the existing data files with a one second interval energy consumption, converts them to 
        a one hour interval, stores the results and deletes the files containing the per second data.

    Args:
        actor_list (Optional[List[str]], optional): the list of actors for which the data needs to be converted. 
                                                    if no list is provided, all files in the HOURS_DIRECTORY (as defined in config.ini) will
                                                    be converted. Defaults to None.
        append (bool, optional): if true the data will be appended to the previous per hour data of that actor. If false, any existing
                                 file for that actor will be overwritten. Defaults to False.
        only_total (bool, optional): if True only the per second total consumption file will be transformed and all per appliance, 
                                     per activity and per user files will be deleted. If False, all files will be converted. Defaults to True.
    """
    # create the destination folder if not exists
    path = os.path.join(os.getcwd(),OUTPUT_PATH,HOURS_DIRECTORY)
    if not os.path.exists(path):
        os.mkdir(path)

    # count the number of folders where per second household consumption files are stored
    if not actor_list:
        actor_list = [
            name for name in os.listdir(os.path.join(OUTPUT_PATH,SECONDS_DIRECTORY)) 
            if not os.path.isfile(os.path.join(OUTPUT_PATH,SECONDS_DIRECTORY, name)) and name.startswith(ACTOR_NAME_PREFIX)]
            

    for actor in actor_list:
        print(f"Resampling the files of {actor}")

        # create new folder for the house if not exists
        output_seconds_directory = os.path.join(os.getcwd(),OUTPUT_PATH,SECONDS_DIRECTORY,actor) 
        output_hour_directory = os.path.join(os.getcwd(),OUTPUT_PATH,HOURS_DIRECTORY,actor) 
        if not os.path.exists(output_hour_directory):
            os.mkdir(output_hour_directory)

        # Find all the potential files to convert in the folder
        files_to_convert = [f for f in os.listdir(output_seconds_directory) 
                            if os.path.isfile(os.path.join(output_seconds_directory, f))]

        for file in files_to_convert: 
            
            # filter out irrelevant files
            if (not only_total and file != "events.csv") or (only_total and file == "total.csv" ):

                # load data in pandas dataframe
                df = pd.read_csv(os.path.join(output_seconds_directory,file), header=None,sep=';')  
                df.columns = ['Date', 'Power']
                data = []
                number_of_rows = int(len(df) / 3600)  # Convert 24*60*60 = 86400 seconds (rows) to 24*60 = 1440 minutes (rows)

                # combine sets of 3600 rows and get mean to convert to hours 
                for x in range(number_of_rows):
                    first = x * 3600
                    last = first + 3600
                    date = df.iloc[first]["Date"]
                    power = df[:last]["Power"].iloc[first:].mean() 
                    data.append([date, power])

                # Creates csv file in output/minute folder
                outname = os.path.join(output_hour_directory, file)
                df = pd.DataFrame(data, columns=['Datetime', 'Energy_Consumed_Wh'])
                append_file = os.path.exists(outname) and append
                mode = 'a' if append_file else 'w'
                df.to_csv(outname, columns=['Datetime','Energy_Consumed_Wh'], sep=";", index=False,mode=mode,header= (not append_file))
    
        # remove the per second consumption files
        for file in files_to_convert:
            os.remove(os.path.join(output_seconds_directory,file))



def load_house_consumption(actor_list:Optional[List[str]]=None) -> Dict[str,pd.DataFrame]:
    """reads in one or more consumption data files and returns it as a pandas dataframe

    Args:
        actor_list (Optional[List[str]], optional): the list of actors for which the data is required. 
                                                    If no list is provided, the data of all actors will be 
                                                    included in the Dataframe. Defaults to None.
    Returns:
        Dict[str,pd.DataFrame]: a dict with the actor names as keys and a dataframe 
                                with their respective calculated energy consumption as values
    """
    
    if not actor_list:
        actor_list = [
            name for name in os.listdir(os.path.join(OUTPUT_PATH,HOURS_DIRECTORY)) 
            if not os.path.isfile(os.path.join(OUTPUT_PATH,HOURS_DIRECTORY, name)) and name.startswith(ACTOR_NAME_PREFIX)]
    
    house_consumption_dict = {}
    for folder in actor_list:
        total_file = [f for f in os.listdir(os.path.join(OUTPUT_PATH,HOURS_DIRECTORY,folder)) if 
                      os.path.isfile(os.path.join(OUTPUT_PATH,HOURS_DIRECTORY,folder, f)) and f == "total.csv"]
        for file in total_file:
            with open(os.path.join(OUTPUT_PATH,HOURS_DIRECTORY,folder,file),"r") as csv_file:
                df = pd.read_csv(csv_file, header=0,sep=';')  
            house_consumption_dict[folder] = df

    return house_consumption_dict



def load_agent_consumption(agent_id:str) -> Dict[str,pd.DataFrame]: 
    """ loads the energy consumption for one specific agent with its corresponding actor data

    Args:
        agent_id (str): the name of the agent. must be of type "agent_XX" where XX is 
                        the index that corresponds to the actor index

    Returns:
        Dict[str,pd.DataFrame]: a dict with the agent_id as key and a dataframe as consumption values
    """
     
    index = agent_id.split("_")[1]
    actor_name = "actor_" + index
    result = load_house_consumption([actor_name])
    return {agent_id:result[actor_name]}