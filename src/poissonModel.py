from common.argumentsChecker import argumentChecker, argumentTypeChecker
from common.modelLibraries import ModelLibraries
import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from math import exp, factorial # funções matemáticas 'exp' e 'factorial'
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação de modelos de contagem
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP,ZeroInflatedPoisson
# pacote acima para a estimação dos modelos ZINB e ZIP, respectivamente
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
# pacote anterior para a realização do teste de Vuong
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from tqdm import tqdm # para mostrar o progresso do loop
from common.modelLibraries import ModelLibraries

class PoissonCanonicalFunction():
    '''
    Class to eval the Poisson canonical function
    '''
    def __repr__(self):
        return f"""
logit = {self.__logitEval:.4f}
chance = {np.exp(self.__logitEval):.4f}
        """
    
    def __str__(self):
        return f"""
logit = {self.__logitEval:.4f}
chance = {np.exp(self.__logitEval):.4f}
        """

    def set_params(self, *params):
        '''
        *params:alpha, beta1, beta2, beta3...
        '''
        self.params = params
        
        return self
    
    def logit(self, *vars):
        '''
        *vars: Observation i of the given predictors.
        '''
        alpha = self.params[0]
        summ = alpha
        for each_p, each_var in zip(self.params[1:], vars):
            summ+=each_p*each_var

        self.__logitEval = summ
        return summ
    

class PoissonModel():


    def __init__(self, generalClass : object, library : ModelLibraries = None, **kwargs) -> None:

            if library == "smf":
                self.model = smf.glm(formula=kwargs.get('formula'),
                                            data=generalClass.dataframe,
                                            family=sm.families.Poisson()).fit()

            # Parâmetros do 'modelo_poisson'
            self.library_used = "smf"

    def showLastResults(self):

        if self.library_used=='smf':
            return self.model.summary()

    def predictLastModel(self, dependentVars: pd.DataFrame):
        '''
        Method to predict the last model. 

        dependentVars :  dataFrame of with the vars to be evaluated. Must exist in the original
        dataframe. 

        Exemple:

        clm.predictLastModel(pd.Dataframe({"distancia":[25], 
                                        "temperatura":[123.43]
                                        }))
        '''
        if self.library_used=='smf':
            return self.model.predict(dependentVars)
