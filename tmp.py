import torch
import numpy as np

b=torch.from_numpy(np.array(1E-20, dtype=np.float32))
a=torch.from_numpy(np.array(1E-46, dtype=np.float32))

c=b/a
print(c)