# %%
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption)

t = dkML.PoissonModel('smf', formula='violations ~ staff + post + corruption')
dkML.showLastResults()

# %%
'''
Estimação para Binomial negativa
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption)
t = dkML.NegBinomial('smf', formula='violations ~ staff + post + corruption', n_samples=100)
dkML.showLastResults()
# %%
'''
Comparação binomial negativa com Poisson
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption) # carregando o dataframe
# estimando o modelo Poisson, com a lib smf
dkML.PoissonModel('smf', formula='violations ~ staff + post + corruption')
poissonModel  = dkML.getLastModel()

# estimando o modelo Binomial negativo, com o lib smf
dkML.NegBinomial('smf', formula='violations ~ staff + post + corruption', n_samples=100)
negativeBiModel  = dkML.getLastModel()

summary_col([poissonModel, negativeBiModel], 
            model_names=["Poisson","BNeg"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
                })

# %%
'''
Comparação binomial negativa com Poisson usando a razão da verossimilhança
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption) # carregando o dataframe
# estimando o modelo Poisson, com a lib smf
dkML.PoissonModel('smf', formula='violations ~ staff + post + corruption')
poissonModel  = dkML.getLastModel()

# estimando o modelo Binomial negativo, com o lib smf
dkML.NegBinomial('smf', formula='violations ~ staff + post + corruption', n_samples=100)
negativeBiModel  = dkML.getLastModel()

dkML.lrtest([poissonModel, negativeBiModel])

# %%
'''
Executando teste de super dispersão:
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption) # carregando o dataframe
# estimando o modelo Poisson, com a lib smf
dkML.PoissonModel('smf', formula='violations ~ staff + post + corruption')
dkML.checkOverDisp()

# %%
'''
Ajustando um modelo ZIP
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption) # carregando o dataframe
# Estimando o modelo ZIP, com a lib smf

dkML.ZeroInflatePoisson('smf', formula='violations ~ staff + post + corruption', comp_logit='corruption')
dkML.getLastModel().summary()
# %%
'''
Executa o teste de Vuong para checar se há a inflação de zeros. Em geral, compara-se o modelo de Poisson com o modelo ZIP

Primeiro, ajusta-se o modelo Poisson, após, ajusta-se o modelo ZIP, e então se faz o teste de Vuong
'''
import sys
sys.path.append('../src')
from main import DanketsuML
import pandas as pd

df_corruption = pd.read_csv('corruption.csv', delimiter=',')
dkML = DanketsuML(df_corruption) # carregando o dataframe
# Ajustando o modelo Poisson:
dkML.PoissonModel('smf', formula='violations ~ staff + post + corruption')
modeloPoisson = dkML.getLastModel()
# Ajustando o modelo ZIP:
dkML.ZeroInflatePoisson('smf', formula='violations ~ staff + post + corruption', comp_logit='corruption')
modeloZip = dkML.getLastModel()
dkML.vuongTest(modeloPoisson, modeloZip)
# %%

