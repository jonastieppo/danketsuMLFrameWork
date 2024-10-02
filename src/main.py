# %%
import pandas as pd
from common.argumentsChecker import argumentChecker, argumentTypeChecker
from common.modelLibraries import ModelLibraries
from poissonModel import PoissonCanonicalFunction, PoissonModel

class DanketsuML():
    '''
    Classe para construir modelos mais facilmente (para eu me organizar)
    '''
    
    def __init__(self, dataframe : pd.DataFrame):

        self.dataframe = dataframe

    def showLastResults(self):
        return self.lastShowMethod()

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
        return self.lastPredictionMethod(dependentVars)

    def PoissonModel(self, library : ModelLibraries = None, **kwargs):
        '''
        Calcultes de Poisson model
        '''
        if library=='smf':
            formula = kwargs.get("formula")
            # check if the arguments passed is valid
            argumentChecker(kwargs, "formula")
            if argumentTypeChecker(str, formula):
                PoissonClass = PoissonModel(self,library,formula=formula)
                self.lastPredictionMethod = PoissonClass.predictLastModel
                self.lastShowMethod = PoissonClass.showLastResults

        return PoissonCanonicalFunction()
    


    
    # def NegBinomial(self, library : ModelLibraries = None, **kwargs):
    #     '''
    #     Fits a model using negative binomial procedure
    #     '''
    #     if library=='smf':
    #         formula = kwargs.get("formula")
    #         formula = kwargs.get("n_samples")

    #         # check if the arguments passed is valid
    #         self.__argumentChecker(kwargs, "formula")
    #         self.__argumentChecker(kwargs, "n_samples")
            
    #         if self.__argumentTypeChecker(str, formula):
    #             self.modelo_poisson = smf.glm(formula=formula,
    #                                     data=self.dataframe,
    #                                     family=sm.families.Poisson()).fit()

    #         # Parâmetros do 'modelo_poisson'
    #         self.last_library = "smf"
    #         self.last_model =  "poisson"
        
    #     return PoissonCanonicalFunction()
# %%
