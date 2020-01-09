'''
generate the 10 minutes rolling data
'''

import data_utility as du
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

input_dir_list = [
	"11/data_preproccessing_10/",
	"12/data_preproccessing_10/"]



merge_traffic = np.empty((8640, 13, 13, 2))

timestep = 0
for input_dir in input_dir_list:
	filelist = du.list_all_input_file(input_dir)
	filelist.sort()
	for datas in filelist:
		data = np.load(os.path.join(input_dir,datas))
		for t in tqdm(range(data.shape[0])):
			index_rol = -1
			for r in range(data.shape[1]):
				if r%8 == 0:
					index_rol += 1	
				index_col = -1	
				for c in range(data.shape[2]):
					if c%8 == 0:
						index_col += 1
					merge_traffic[timestep, index_rol, index_col, 0] = int(data[t,r,c,1])
					merge_traffic[timestep, index_rol, index_col, 1] += int(data[t,r,c,6])
			timestep += 1
np.save("npy_merge/merge_traffic", merge_traffic)