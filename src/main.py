# %%
import pandas as pd
from binomialNegativeModel import BinomialCanonicalFunction, BinomialNegativeModel
from common.argumentsChecker import argumentChecker, argumentTypeChecker
from common.modelLibraries import ModelLibraries
from common.overdisp import OverDisp
from poissonModel import PoissonCanonicalFunction, PoissonModel
from lrTest import lrTest
from zeroInflatedPoission import ZeroInflatedPoisson
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
        return self.makePredictionLastModel(dependentVars)
    
    def getLastModel(self):
        '''
        Method to acess last used Model
        '''
        return self.lastUsedModel

    def PoissonModel(self, library : ModelLibraries = None, **kwargs):
        '''
        Calcultes de Poisson model
        '''
        self.library = library
        self.modelkwArgs = kwargs
        PoissonClass = PoissonModel(self)
        self.makePredictionLastModel = PoissonClass.predictLastModel
        self.lastShowMethod = PoissonClass.showLastResults
        self.lastUsedModel = PoissonClass.model

        return PoissonCanonicalFunction()
    
    
    def NegBinomial(self, library : ModelLibraries = None, **kwargs):
        '''
        Fits a model using negative binomial procedure
        '''
        self.library = library
        self.modelkwArgs = kwargs
        BiNegClass = BinomialNegativeModel(self)
        self.makePredictionLastModel = BiNegClass.predictLastModel
        self.lastShowMethod = BiNegClass.showLastResults
        self.lastUsedModel = BiNegClass.model

        return BinomialCanonicalFunction()
    
    def ZeroInflatePoisson(self, library : ModelLibraries = None, **kwargs):
        '''
        Implementação feita sobre algumas implementações já disponíveis para
        o modelo Zero-Inflated Poisson

        LAMBERT, D. Zero-inflated Poisson regression, with an application to defects
        in manufacturing. Technometrics, v. 34, n. 1, p. 1-14, 1992.
        '''
        self.library = library
        self.modelkwArgs = kwargs
        ZIPModel = ZeroInflatedPoisson(self)
        self.makePredictionLastModel = ZIPModel.predictLastModel
        self.lastShowMethod = ZIPModel.showLastResults
        self.lastUsedModel = ZIPModel.model
    
    def lrtest(self,models):
        '''
        Executa o teste da razão da verossimilhança
        '''
        if not self.library:
            raise Exception("Please, perform a prediction first")
        
        lrTest(models=models, library=self.library)

    def checkOverDisp(self):
        '''
        Executa o teste de superdispersão (funciona apenas para a lib smf)
        '''
        formula = self.modelkwArgs["formula"]
        return OverDisp(self, formula=formula)


# %%
