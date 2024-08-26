""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
This File contains a script to generate a file that contains one or more 
energy production actor profiles and the randomly generated characteristics of their photovoltaic installation.

The script uses some predetermined configuration variables stored in the config.ini file.

To use the script run:
'python data_synthetic_preprocess/production_create_profiles.py -n <N> -i <I> -a'
-n <N> where <N> is the number of actors for which to create a profile
-i <I> where <I> is the starting index for the first actor.
-a when -a is added the generated profiles will be appended to the existing file instead of creating a new file.

"""

import argparse
import configparser
import os

import production_helpers as ph

if __name__ == "__main__":
    
    # read the configuration settings from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))
    output_path = os.path.join(os.getcwd(),config["SYNTHETIC_DATA"]["synthetic_data_folder"])
    outname = os.path.join(output_path, config['SYNTHETIC_DATA']['production_actors_filename'])

    # process the command line arguments
    parser = argparse.ArgumentParser(prog='Production actor creator',description='Create new random production actor profiles')
    parser.add_argument('-n', metavar="X", type=int, required=True, dest="nbr_of_actors", help="the number of actor profiles to create")
    parser.add_argument('-i', metavar="X", type=int, default=0, dest="start_nbr_of_actor", help="the index for the first actor")
    parser.add_argument('-a', dest="append",action="store_true", help="appends the new actors to the exisitng file (if one exists) instead of creating a new file",)
    args = parser.parse_args()

    nbr_of_actors = args.nbr_of_actors
    start_nbr_of_actor = args.start_nbr_of_actor
    
    file_exists = os.path.exists(outname)

    # print overview of what will be generated in terminal
    os.system('cls' if os.name == 'nt' else 'clear')  # noqa: S605 
    print("-" *100)
    if file_exists and args.append:
        print(str.upper("This script will append new actor production configs for:"))
        for i in range(start_nbr_of_actor,start_nbr_of_actor + nbr_of_actors):
            print(f"\t{ph.ACTOR_NAME_PREFIX}_{i:02d}")
        print(f"in file {outname}.csv")
    else:
        print(str.upper("This script will create new actor production configs for:"))
        for i in range(start_nbr_of_actor,start_nbr_of_actor + nbr_of_actors):
            print(f"\t{ph.ACTOR_NAME_PREFIX}_{i:02d}")
        print(f"\nin file {outname}.csv")
    print("-" *100)
    
    # request confirmation from user
    validate = input("\nAre you sure you want to continue [YES/NO]:")
    if validate == "YES":
        # create profiles
        actors = ph.create_random_profiles(nbr_of_actors,start_nbr_of_actor)
        ph.save_profiles_to_file(actors,args.append)

    else:
        print("Program terminated by user.")

