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
from templates.templates import DanketsuTemplate
from tqdm import tqdm # para mostrar o progresso do loop


class BinomialCanonicalFunction():
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
    

class BinomialNegativeModel():


    def __init__(self, generalClass : DanketsuTemplate) -> None:

        if generalClass.library=='smf':
            formula = generalClass.modelkwArgs.get("formula")
            n_samples = generalClass.modelkwArgs.get("n_samples")
            # check if the arguments passed is valid
            argumentChecker(generalClass.modelkwArgs, "formula")
            argumentChecker(generalClass.modelkwArgs, "n_samples")
            # check the type of the arguments
            if bool(argumentTypeChecker(str, formula)*argumentTypeChecker(int,n_samples)):
                
                # Encontra o Phi otimo
                best_phi = self.__findBestPhi(generalClass.dataframe, formula=formula,n_samples=n_samples)
                self.model = smf.glm(formula=formula,
                                            data=generalClass.dataframe,
                                            family=sm.families.NegativeBinomial(alpha=best_phi)).fit()

                self.model.CustomModelName = "Binomial Negative"

            self.library_used = "smf"

    def __findBestPhi(self, dataframe : pd.DataFrame, formula : str, n_samples : int):
        '''
        Private method to find the best phi, based on a quantity of samples
        '''
        alphas = np.linspace(0, 10, n_samples)
        llf = np.full(n_samples, fill_value=np.nan)

        for i, alpha in tqdm(enumerate(alphas), total=n_samples, desc='Estimating'):
            try:
                model = smf.glm(formula=formula,
                                data=dataframe,
                                family=sm.families.NegativeBinomial(alpha=alpha)).fit()
            except:
                continue
            llf[i] = model.llf

        return alphas[np.nanargmax(llf)]


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
