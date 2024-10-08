from templates.templates import SMFTemplate
from scipy import stats

class lrTest():
    '''
    Class to perform the LR test
    '''
    
    def __init__(self, models : list, library : str) -> None:
        
        if library=='smf':
            self.__lrtest_SMF(modelos=models)
            return 
        

        # If is passed some non-existend library:
        raise Exception(f"Biblioteca {library} não-existente")
    
    def __lrtest_SMF(self, modelos: list[SMFTemplate]):
        '''
        Perfomrs the comparison between two models, when they are fitted using
        the smf library. In portuguese: "Teste da razão de verosimilhança
        '''

        modelo_1 = modelos[0]
        llk_1 = modelo_1.llnull
        llk_2 = modelo_1.llf

        if len(modelos)>1:
            llk_1 = modelo_1.llf
            llk_2 = modelos[1].llf
        LR_statistic = -2*(llk_1-llk_2)
        p_val = stats.chi2.sf(LR_statistic, 1) # 1 grau de liberdade

        print("Likelihood Ratio Test:")
        print(f"-2.(LL0-LLm): {round(LR_statistic, 2)}")
        print(f"p-value: {p_val:.3f}")
        print("")
        print("==================Result======================== \n")

        # Salvando o modelo com maior likely hood
        if llk_1>llk_2:
            best_model = modelo_1

        else:
            best_model = modelos[1]

        if p_val <= 0.05:
            print(fr'''
*--------------------------------------------------------------------*

H1: Different models, favoring the one with the highest Log-Likelihood
So, the best model is {best_model.CustomModelName}

*--------------------------------------------------------------------*
''')
        else:
            print(r"H0: Models with log-likelihoods that are not statistically different at 95% confidence level")