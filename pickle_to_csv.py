#!/usr/bin/env

import os
import pandas as pd

for filename in os.listdir("processed/"):
    try:
        d = pd.read_pickle("processed/"+filename)
        d.to_csv("processed/"+filename.replace('pkl','csv'))
        print("processed/"+filename.replace('pkl','csv'))
    except:
        print("faile to process"+filename)
    
    