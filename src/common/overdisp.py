from templates.templates import DanketsuTemplate
import statsmodels.formula.api as smf # estimação de modelos de contagem
import statsmodels.api as sm # estimação de modelos

class OverDisp():
    '''
    Class to check the over dispersion
    '''
    def __init__(self, generalClass : DanketsuTemplate, formula : str) -> None:
        
        model = generalClass.getLastModel()
        df = generalClass.dataframe
        df['lambda_poisson'] = model.fittedvalues
        y_var = formula.split(sep=' ~ ')[0]
        
        # Criando a nova variável Y* ('ystar')
        df['ystar'] = (((df['y_var']
                                    -df['lambda_poisson'])**2)
                                -df['y_var'])/df['lambda_poisson']

        # Estimando o modelo auxiliar OLS, sem o intercepto
        modelo_auxiliar = sm.OLS.from_formula('ystar ~ 0 + lambda_poisson',
                                            df).fit()

        # Parâmetros do 'modelo_auxiliar'
        modelo_auxiliar.summary()