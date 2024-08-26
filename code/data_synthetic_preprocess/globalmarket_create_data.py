""" 
-------------------------------------------------------------------------------------------
A JAX FRAMEWORK FOR MULTI-AGENT RESEARCH IN DECENTRALISED ENERGY MARKETS
-------------------------------------------------------------------------------------------
author: K. Emerson
Department of Computer Science
Faculty of Sciences and Bioengineering Sciences
Vrije Universiteit Brussel
-------------------------------------------------------------------------------------------
file contains a script to fetch real historic day ahead energy market prices per Wh and store them.

The script uses the entsoe-py library to fetch day ahead energy market prices from the 
"european network of transmission system operators for electricity" transparancy platform.
(for more info, please consult: https://github.com/EnergieID/entsoe-py)

The script uses some predetermined configuration variables stored in the config.ini file.
To use the script run:
'python data_synthetic_preprocess/globalmarket_create_data.py -a <A> -s <S> -e <E> -f <F>'
The two command line arguments do the following:
-a <A> where <A> is the API-key to use (required)
-s <S> where <S> is the startdate from where the prices should be fetched, if not provided the default in config.ini will be used
-e <E> where <E> is the enddate upon which the prices should be fetched, if not provided the default in config.ini will be used
-f <F> where <F> is the filename to use to store the data, if not provided "globalmarket.csv" will be used
"""

import argparse
import configparser
import os

import pandas as pd
from entsoe import EntsoePandasClient

if __name__ == "__main__":

    # get all the configurations from config.ini
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(),'data_synthetic_preprocess','config.ini'))
    output_path = os.path.join(os.getcwd(),config['SYNTHETIC_DATA']['synthetic_data_folder'])
    output_directory = os.path.join(output_path,config['SYNTHETIC_DATA']['global_market_subfolder'])
    timezone = config['SYNTHETIC_DATA.LOCALISATION']['timezone']
    country_code = config['SYNTHETIC_DATA.LOCALISATION']['country_code']
    default_startdate = config["SYNTHETIC_DATA"]["startdate"]
    default_enddate = config["SYNTHETIC_DATA"]["enddate"]

    # process the command line arguments
    parser = argparse.ArgumentParser(prog='Global Market Data creator',description='Creates a new global market data file based on true values')
    parser.add_argument('-a', metavar="X", dest="api_key", help="the API key to use",required=True)
    parser.add_argument('-s', metavar="YYYY-MM-DD", default=default_startdate, dest="startdate", help="the starting date")
    parser.add_argument('-e', metavar="YYYY-MM-DD", default=default_enddate, dest="enddate", help="the end date")
    parser.add_argument('-f', metavar="X", default="globalmarket.csv", dest="filename", help="the filename to use")
    args = parser.parse_args()
    
    startdate = args.startdate
    enddate = args.enddate
    api_key = args.api_key 
    filename = args.filename
       
    # print overview of what will happen in terminal
    os.system('cls' if os.name == 'nt' else 'clear')  # noqa: S605
    print("-" *150)
    print(str.upper("This script will create new Global market data in:"))
    
    outname = os.path.join(output_path, filename)
    print(f"\t{outname}")
    print("-" *150)
    validate = input("Are you sure you want to continue [YES/NO]:")
    
    # request confirmation from user
    if validate == "YES":
        
        # create client
        client = EntsoePandasClient(api_key=api_key)
        start = pd.Timestamp(startdate, tz=timezone)
        end = pd.Timestamp(enddate, tz=timezone)
        print("fetching data from api")
        
        # query the day ahead prices
        result = client.query_day_ahead_prices(country_code, start=start, end=end)
        
        # transform and store the data
        result = pd.DataFrame({'Date':result.index, 'Day_ahead_price':result.values})
        result["Datetime"] = pd.to_datetime(result["Date"],yearfirst=True,utc=True).dt.tz_convert(timezone).dt.tz_localize(None)
        result['Day_ahead_price_Wh'] = result['Day_ahead_price']/(1000 * 1000) # from MWh to Wh
        result = result.drop(columns=['Day_ahead_price']) 
        
        result = result.drop(result.tail(1).index)

        # save file
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        print("storing data from api")
        outname = os.path.join(output_directory, filename)
        with open(outname, "w") as csv_file:
            result.to_csv(csv_file, columns=["Datetime", "Day_ahead_price_Wh"], sep=";", index=False)
    
    else:
        print("Program terminated by user.")