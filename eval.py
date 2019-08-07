import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import read_target

df = pd.read_csv('./result.csv', index_col=0)
print(df.mean())
