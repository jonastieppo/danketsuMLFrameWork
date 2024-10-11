import pandas as pd
from common.argumentsChecker import argumentChecker, argumentTypeChecker
from templates.templates import DanketsuTemplate
import statsmodels.api as sm # estimação de modelos

class ZeroInflatedPoisson():
    '''
    Classe para utilizar o modelo de Poisson Inflacionado por Zeros
    '''
    def __init__(self, generalClass : DanketsuTemplate) -> None:

        if generalClass.library == 'smf':
            formula = generalClass.modelkwArgs.get("formula")
            comp_logit = generalClass.modelkwArgs.get("comp_logit")
            # check if the arguments passed is valid
            argumentChecker(generalClass.modelkwArgs, "formula")
            argumentChecker(generalClass.modelkwArgs, "comp_logit")
        if bool(argumentTypeChecker(str, formula)*argumentTypeChecker(str,comp_logit)):
        pass

    def __smfDummiesBugWrap(self, dataframe : pd.Dataframe, formula :  str, comp_logit : str):
        '''
        É necessário dummizar as variáveis categóricas ao 
        se utilizar o ZIP, se não ocorre um erro.

        Para tal:

        1) Parsa a string referente ao modelo utilizado
        2) Para todas as colunas do tipo objeto, utilzar o pd.get_dummies, 
        dropando a primeira
        3) Ajusta o modelo normalmente.
        '''
        y_column = formula.split(' ~ ')[0]
        first_var = formula.split(' ~ ')[1]
        first_var = first_var.split(' + ')[0]
        others_vars = formula.split(' + ')[1:]
        preditors = first_var + others_vars
        # Definição da variável dependente

        y = dataframe[y_column]

        # definição das variáveis preditoras que entração no modelo de contagem (poisson)

        x = dataframe[preditors]
        x_with_const = sm.add_constant(x)
        object_columns = dataframe.columns[dataframe.dtypes.values=='object']
        
        # dummizar todas as colunas de string
        cols_to_dummy = []
        for each_var in preditors:
            if each_var in object_columns:
                cols_to_dummy.append(cols_to_dummy)

        x_final_poisson = pd.get_dummies(x_with_const, columns=[cols_to_dummy], dtype=int, drop_first=True)

        # Definição das variáveis preditoras que entração no logit
        
