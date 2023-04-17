import pandas as pd
import torch
import random
import os


df = pd.read_excel("datasets/Haberman's_Survial_Data/Q2-Haberman's Survival Data.xlsx", sheet_name='Sheet1')
df_X = df.loc[:, ['Age', 'year', 'Number of positive axillary nodes detected']]
df_y = df.loc[:, ['Survival status']]
X = torch.tensor(df_X.values)
y_ = torch.tensor(df_y.values)
y = torch.zeros((len(y_), 2))
for i in range(len(y_)):
    y[i, y_[i]-1] = 1
training_indices = random.sample(range(len(X)), 200)
test_indices = list(set(range(len(X))) - set(training_indices))
training_X = X[training_indices].to(torch.float)
training_y = y[training_indices].to(torch.float)
test_X = X[test_indices].to(torch.float)
test_y = y[test_indices].to(torch.float)

save_dir = "datasets/Haberman's_Survial_Data"
os.makedirs(save_dir, exist_ok=True)
torch.save(training_X, os.path.join(save_dir, "training_X.pt"))
torch.save(training_y, os.path.join(save_dir, "training_y.pt"))
torch.save(test_X, os.path.join(save_dir, "test_X.pt"))
torch.save(test_y, os.path.join(save_dir, "test_y.pt"))
print(f"Saving datasets into [{save_dir}], please check them!")