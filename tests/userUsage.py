# %%
import sys
sys.path.append('../src')
from main import CleanModels

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
classInit = CleanModels(df_corruption)

classInit.PoissonModel('smf', formula='violations ~ staff + post + corruption')
classInit.showLastResults()   
classInit.predictLastModel(pd.DataFrame({'staff':[23],
                                     'post':['no'],
                                     'corruption':[0.5]}))   

p_model = classInit.PoissonModel().set_params(0.1,1,2)