# %%
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
classInit = DanketsuML(df_corruption)

t = classInit.PoissonModel('smf', formula='violations ~ staff + post + corruption')
classInit.showLastResults()
# classInit.predictLastModel(pd.DataFrame({'staff':[23],

#                                      'post':['no'],
#                                      'corruption':[0.5]}))   

# p_model = classInit.PoissonModel().set_params(2.0963,

# p_model.logit(1,10,11)

# p_model
# %%
p_model = classInit.PoissonModel().set_params(1.9469,-4.2746,0.0400,0.4527)

p_model.logit(0,28,1)

p_model


# %%
