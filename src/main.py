# %%

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


class ModelLibraries():

    smf = None

    def __init__(self) -> None:
        self.smf = None

    def __print__(self):
        print("smf")

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
class CleanModels():
    '''
    Classe para construir modelos mais facilmente (para eu me organizar)
    '''
    
    def __init__(self, dataframe : pd.DataFrame):

        self.dataframe = dataframe

    def __argumentTypeChecker(self, expectedType, receivedType):
        '''
        check if the argument is valid
        '''
        if type(receivedType) != expectedType:
            raise Exception(f"Argument invalid. Received {type(receivedType)}, but expected {expectedType}")
        else:
            return True
        
    def __argumentChecker(self, poolOfArguments : dict, expectedArgument):
        '''
        Checks if the method received the correct arguments. It will print the expeted ones
        '''

        def print_args():
            stream = "\n--- Arguments Received----\n"
            for each_arg in poolOfArguments.keys():
                stream+=f"* {each_arg}\n"
            return stream

        if expectedArgument not in poolOfArguments.keys():
            raise Exception(fr"Expeted argument {expectedArgument}, but received: {print_args()}")
        
    def showLastResults(self):

        if self.last_library=='smf' and self.last_model == 'poisson':
            return self.modelo_poisson.summary()

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
        if self.last_library=='smf' and self.last_model == 'poisson':
            return self.modelo_poisson.predict(dependentVars)
        

    def PoissonModel(self, library : ModelLibraries = None, **kwargs):
        '''
        Calcultes de Poisson model
        '''
        
        if library=='smf':
            formula = kwargs.get("formula")

            # check if the arguments passed is valid
            self.__argumentChecker(kwargs, "formula")
            if self.__argumentTypeChecker(str, formula):
                self.modelo_poisson = smf.glm(formula=formula,
                                        data=self.dataframe,
                                        family=sm.families.Poisson()).fit()

            # Parâmetros do 'modelo_poisson'
            self.last_library = "smf"
            self.last_model =  "poisson"
        
        return PoissonCanonicalFunction()
    
    def NegBinomial(self, library : ModelLibraries = None, **kwargs):
        '''
        Fits a model using negative binomial procedure
        '''
        if library=='smf':
            formula = kwargs.get("formula")
            formula = kwargs.get("n_samples")

            # check if the arguments passed is valid
            self.__argumentChecker(kwargs, "formula")
            self.__argumentChecker(kwargs, "n_samples")
            
            if self.__argumentTypeChecker(str, formula):
                self.modelo_poisson = smf.glm(formula=formula,
                                        data=self.dataframe,
                                        family=sm.families.Poisson()).fit()

            # Parâmetros do 'modelo_poisson'
            self.last_library = "smf"
            self.last_model =  "poisson"
        
        return PoissonCanonicalFunction()

# %%
