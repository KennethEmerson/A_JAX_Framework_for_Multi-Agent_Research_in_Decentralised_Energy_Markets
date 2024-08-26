""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------

File contains the script to create new randomly generated consumption household profile files
and uses the predetermined configuration variables stored in the config.ini file.

Each generated consumption household profile file contains a chosen baseload and a
list of users, which on their terms each have specific predefined consumption profiles.

The resulting profiles can be used to generate synthetic consumption data for one specific
household using the script in the file 'consumption_create_data.py' and the ANTgen library.
(For more info on ANTgen, please consult: https://gitlab.com/nunovelosa/antgen)

The script uses some predetermined configuration variables stored in the config.ini file.
To use the script, run:
'python data_synthetic_preprocess/consumption_create_profiles.py -n <N> -s <S>'
The two command line arguments do the following:
-n <N> where <N> is the number of consumption profiles to create
-s <S> where <S> is the starting index for the first profile, which is used to determine
the filename where the config is stored. if not provided the starting index is zero
"""

import argparse
import configparser
import os

import consumption_helpers as ch

if __name__ == "__main__":
    
    # read the configuration settings from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))
    max_residents = int(config["SYNTHETIC_DATA.CONSUMPTION"]["max_residents_per_actor_house"])
    output_path = os.path.join(os.getcwd(),config["SYNTHETIC_DATA"]["synthetic_data_folder"])
    date_format = "%Y-%m-%d"

    # process the command line arguments
    parser = argparse.ArgumentParser(prog='Consumption agent creator',description='Create new random consumption agents')
    parser.add_argument('-n', metavar="X", type=int, required=True, dest="nbr_of_agents", help="the number of agents to create")
    parser.add_argument('-s', metavar="X", type=int, default=0, dest="start_nbr_of_agent", help="the number for the first agent")
    args = parser.parse_args()
    start_agent = args.start_nbr_of_agent
    nbr_of_agents = args.nbr_of_agents
       
    
    os.system('cls' if os.name == 'nt' else 'clear')  # noqa: S605
    print("-" *150)
    print(str.upper("This script will create new agent consumption config files in:"))
    for i in range(start_agent,start_agent + nbr_of_agents):
        print(f"\t{output_path}/{ch.AGENTS_DIRECTORY}/{ch.ACTOR_NAME_PREFIX}_{i:02d}")
    print("\n")
    print("-" *150)
    validate = input("Are you sure you want to continue [YES/NO]:")
    
    # request confirmation from user
    if validate == "YES":
        # create the configuration files   
        agent_list = ch.create_random_houses(nbr_of_houses=nbr_of_agents, start_nbr_house=start_agent, max_residents=max_residents)
    
    else:
        print("Program terminated by user.")
    