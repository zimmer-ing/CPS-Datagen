#!/usr/bin/env python3
# %%
from pathlib import Path
import time
import itertools
import virtualCPPS as vCPPS


start_time = time.time()
parameters=vCPPS.gen_parameters.cross_param()


for config in parameters:
    print("Generating data collection:", config.collection_name)
    dataset_collection = vCPPS.DataSetCollection.generate(config, config.collection_name)


print("Time taken:", time.time() - start_time)
