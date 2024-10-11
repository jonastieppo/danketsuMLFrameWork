from templates.templates import DanketsuTemplate
import statsmodels.formula.api as smf # estimação de modelos de contagem
import statsmodels.api as sm # estimação de modelos

class OverDisp():
    '''
    Class to check the over dispersion
    '''

    def __str__(self):
    
        return self.summary
    
    def __repr__(self) -> str:
         
        return str(self.summary)
    
    def __init__(self, generalClass : DanketsuTemplate, formula : str) -> None:
        
        model = generalClass.getLastModel()
        df = generalClass.dataframe
        df['lambda_poisson'] = model.fittedvalues
        y_var = formula.split(sep=' ~ ')[0]
        
        # Criando a nova variável Y* ('ystar')
        df['ystar'] = (((df[y_var]
                                    -df['lambda_poisson'])**2)
                                -df[y_var])/df['lambda_poisson']

        # Estimando o modelo auxiliar OLS, sem o intercepto
        modelo_auxiliar = sm.OLS.from_formula('ystar ~ 0 + lambda_poisson',
                                            df).fit()

        # Parâmetros do 'modelo_auxiliar'
        self.summary = modelo_auxiliar.summary()

        pvalue = modelo_auxiliar.pvalues.values[0]

        if pvalue>0.05:
            print(fr'''
============================================================================
                                RESULTADO:
O teste de superdispersão de Cameron e Trivedi resultadou em um p-value de {pvalue:.4f}, que é 
maior que alpha=0.05. Logo, o modelo auxiliar não é significativo estatisticamente.
Portanto, o modelo tem EQUIDISPERSÃO. Ou seja, NÃO tem superdispersão!

==============================================================================
''')
            
        else:
            print(fr'''
============================================================================
                                RESULTADO:
O teste de superdispersão de Cameron e Trivedi resultadou em um p-value de {pvalue:.4f}, que é 
menor que alpha=0.05. Logo, HÁ superdispersão!

==============================================================================
''')

