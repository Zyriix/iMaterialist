import torch
import numpy as np
import pandas as pd
a = torch.Tensor([[0,1,1],
                  [1,1,1]])

b = torch.Tensor([1,2,3])

# print(np.log(0.000000000000000001))

df = pd.read_csv("G:\OneDrive\code\python\maskrcnn-benchmark/result/2020-04-26--rs.csv")
print(df['ImageId'].value_counts())