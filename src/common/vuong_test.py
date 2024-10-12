import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP,ZeroInflatedPoisson
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson

class VuongTest():
    '''
    CLasse criada para executar o teste de Voung. 

    VUONG, Q. H. Likelihood ratio tests for model selection and non-nested
    hypotheses. Econometrica, v. 57, n. 2, p. 307-333, 1989.

    Definição de função para elaboração do teste de Vuong

    Baseada na implementação de:
    # Autores: Luiz Paulo Fávero e Helder Prado Santos
    '''

    def __repr__(self) -> str:
        return self.__makeConclusions()

    def __str__(self) -> str:
        return self.__makeConclusions()

    def __init__(self, model1, model2, pvalue_lim=0.05) -> None:
        
        self.pvalue_lim = pvalue_lim
        self.__vuong_test(model1,model2)
        pass


    def __vuong_test(self,m1, m2): 


        if m1.__class__.__name__ == "GLMResultsWrapper":
            
            glm_family = m1.model.family

            X = pd.DataFrame(data=m1.model.exog, columns=m1.model.exog_names)
            y = pd.Series(m1.model.endog, name=m1.model.endog_names)

            if glm_family.__class__.__name__ == "Poisson":
                m1 = Poisson(endog=y, exog=X).fit()
                
            if glm_family.__class__.__name__ == "NegativeBinomial":
                m1 = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()

        else:
            glm_family = m2.model.family

            X = pd.DataFrame(data=m2.model.exog, columns=m2.model.exog_names)
            y = pd.Series(m2.model.endog, name=m2.model.endog_names)

            if glm_family.__class__.__name__ == "Poisson":
                m2 = Poisson(endog=y, exog=X).fit()
                
            if glm_family.__class__.__name__ == "NegativeBinomial":
                m2 = NegativeBinomial(endog=y, exog=X, loglike_method='nb2').fit()


        supported_models = [ZeroInflatedPoisson,ZeroInflatedNegativeBinomialP,Poisson,NegativeBinomial]
    
        if type(m1.model) not in supported_models:
            raise ValueError(f"Model type not supported for first parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
            
        if type(m2.model) not in supported_models:
            raise ValueError(f"Model type not supported for second parameter. List of supported models: (ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, Poisson, NegativeBinomial) from statsmodels discrete collection.")
                
        # Extração das variáveis dependentes dos modelos
        m1_y = m1.model.endog
        m2_y = m2.model.endog

        m1_n = len(m1_y)
        m2_n = len(m2_y)

        if m1_n == 0 or m2_n == 0:
            raise ValueError("Could not extract dependent variables from models.")

        if m1_n != m2_n:
            raise ValueError("Models appear to have different numbers of observations.\n"
                            f"Model 1 has {m1_n} observations.\n"
                            f"Model 2 has {m2_n} observations.")

        if np.any(m1_y != m2_y):
            raise ValueError("Models appear to have different values on dependent variables.")
            
        m1_linpred = pd.DataFrame(m1.predict(which="prob"))
        m2_linpred = pd.DataFrame(m2.predict(which="prob"))        

        m1_probs = np.repeat(np.nan, m1_n)
        m2_probs = np.repeat(np.nan, m2_n)

        which_col_m1 = [list(m1_linpred.columns).index(x) if x in list(m1_linpred.columns) else None for x in m1_y]    
        which_col_m2 = [list(m2_linpred.columns).index(x) if x in list(m2_linpred.columns) else None for x in m2_y]

        for i, v in enumerate(m1_probs):
            m1_probs[i] = m1_linpred.iloc[i, which_col_m1[i]]

        for i, v in enumerate(m2_probs):
            m2_probs[i] = m2_linpred.iloc[i, which_col_m2[i]]

        lm1p = np.log(m1_probs)
        lm2p = np.log(m2_probs)

        m = lm1p - lm2p

        self.v = np.sum(m) / (np.std(m) * np.sqrt(len(m)))

        self.pval = 1 - norm.cdf(self.v) if self.v > 0 else norm.cdf(self.v)


    def __makeConclusions(self)->str:
        '''
        Executar o teste estatística e estabelecer conclusões
        '''

        if self.pval < self.pvalue_lim:
            return fr'''
================================================
Vuong Non-Nested Hypothesis Test-Statistic (Raw):
Vuong z-statistic: {round(self.v, 3)}

p-value: {self.pval:.3f}

==================Result======================== 
H1: Indicates inflation of zeros at {(1-self.pvalue_lim)*100.:.2f}% confidence level

'''
        else:
            return fr'''
================================================
Vuong Non-Nested Hypothesis Test-Statistic (Raw):
Vuong z-statistic: {round(self.v, 3)}

p-value: {self.pval:.3f}

==================Result======================== 
H0: Indicates no inflation of zeros at {(1-self.pvalue_lim)*100.:.2f}% confidence level
===================================================================================
'''