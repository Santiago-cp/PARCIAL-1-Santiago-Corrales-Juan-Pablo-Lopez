import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
import re
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
import warnings
warnings.filterwarnings("ignore")
from pycaret.classification import *

class ML_FLOW_CLASS:
    def __init__(self):
        self.path = 'C:/Users/totoy/Documents/UNIVERSIDAD/PYTHON/PARCIAL 1/'  
        self.prueba = None
        self.numericas = None
        self.categoricas = None
        self.cuadrado = None
        self.result = None
        self.result2 = None
        self.result3 = None

    def load_data(self):
        df = pd.read_csv(self.path + 'train.csv')
        self.prueba = pd.read_csv(self.path + "test.csv")
        
        df["Target"] = df["Target"].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
        ct = ['Gender', 'Displaced', 'Educational special needs', 'Debtor', 'Scholarship holder', 
          'International']

        for k in ct:
            df[k] = df[k].astype("object")
            self.prueba[k] = self.prueba[k].astype("object")
            
        formato = pd.DataFrame(df.dtypes).reset_index()
        formato.columns = ["Variable", "Formato"]
        self.categoricas = list(formato.loc[formato["Formato"] == "object", "Variable"])
        self.categoricas = [x for x in self.categoricas if x not in ["Target"]]
        self.numericas = list(formato.loc[formato["Formato"] != "object", "Variable"])
        self.numericas = [x for x in self.numericas if x not in ["Target", "id"]]

        return self.categoricas, self.numericas, df

    def load_dataing(self): 
        
        df = pd.read_csv(self.path  + 'train.csv')
        self.prueba = pd.read_csv(path + "test.csv")
        
        df["Target"] = df["Target"].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
        ct = ['Gender', 'Displaced', 'Educational special needs', 'Debtor', 'Scholarship holder', 
          'International']
    
        for k in ct:
            df[k] = df[k].astype("object")
            prueba[k] = prueba[k].astype("object")
            
        formato = pd.DataFrame(df.dtypes).reset_index()
        formato.columns = ["Variable","Formato"]
        categoricas = list(formato.loc[formato["Formato"] == "object",] ["Variable"])
        categoricas = [x for x in categoricas if x not in ["Target"]]
        numericas = list(formato.loc[formato["Formato"]!= "object",] ["Variable"])
        numericas = [x for x in numericas if x not in ["Target", "id"]]
    
        # Variables al cuadrado 
        base_cuadrado = df.get(numericas).copy()
        base_cuadrado["Target"] = df["Target"].copy()
    
        var_names2, pvalue1 = [], []
    
        for k in numericas:
            base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2
    
            # Prueba de Kruskal sin logaritmo
            mue1 = base_cuadrado.loc[base_cuadrado["Target"]==0,k+"_2"].to_numpy()
            mue2 = base_cuadrado.loc[base_cuadrado["Target"]==1,k+"_2"].to_numpy()
            mue3 = base_cuadrado.loc[base_cuadrado["Target"]==2,k+"_2"].to_numpy()
        
            p1 = stats.kruskal(mue1,mue2, mue3)[1]
        
            # Guardar p values y variables
            var_names2.append(k+"_2")
            pvalue1.append(np.round(p1,2))
    
    
        pcuadrado1 = pd.DataFrame({'Variable2':var_names2,'p value':pvalue1})
        pcuadrado1["criterio"] = pcuadrado1.apply(lambda row: 1 if row["p value"]<=0.10 else 0,axis = 1)
        
        # interacciones cuantitavias
        lista_inter = list(combinations(numericas,2))
        base_interacciones = df.get(numericas).copy()
        var_interaccion, pv1 = [], []
        base_interacciones["Target"] = df["Target"].copy()
    
        for k in lista_inter:
            base_interacciones[k[0]+"__"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]
    
            # Prueba de Kruskal
            mue1 = base_interacciones.loc[base_interacciones["Target"]==0,k[0]+"__"+k[1]].to_numpy()
            mue2 = base_interacciones.loc[base_interacciones["Target"]==1,k[0]+"__"+k[1]].to_numpy()
            mue3 = base_interacciones.loc[base_interacciones["Target"]==2,k[0]+"__"+k[1]].to_numpy()
        
            p1 = stats.kruskal(mue1,mue2, mue3)[1]
        
            var_interaccion.append(k[0]+"__"+k[1])
            pv1.append(np.round(p1,2))
    
        pxy = pd.DataFrame({'Variable':var_interaccion,'p value':pv1})
        pxy["criterio"] = pxy.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
    
        # Razones
    
        raz1 = [(x,y) for x in numericas for y in numericas]
        base_razones1 = df.get(numericas).copy()
        base_razones1["Target"] = df["Target"].copy()
    
        var_nm, pval = [], []
        for j in raz1:
            if j[0]!=j[1]:
                base_razones1[j[0]+"__coc__"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]]+0.01)
    
            # Prueba de Kruskal
                mue1 = base_razones1.loc[base_razones1["Target"]==0,j[0]+"__coc__"+j[1]].to_numpy()
                mue2 = base_razones1.loc[base_razones1["Target"]==1,j[0]+"__coc__"+j[1]].to_numpy()
                mue3 = base_razones1.loc[base_razones1["Target"]==2,j[0]+"__coc__"+j[1]].to_numpy()
                p1 = stats.kruskal(mue1,mue2, mue3)[1]
            
            # Guardar valores
                var_nm.append(j[0]+"__coc__"+j[1])
                pval.append(np.round(p1,2))
    
        prazones = pd.DataFrame({'Variable':var_nm,'p value':pval})
        prazones["criterio"] = prazones.apply(lambda row: 1 if row["p value"]<=0.10 else 0, axis = 1)
    
        # interacciones categoricas
    
        categoricas = list(formato.loc[formato["Formato"]=="O","Variable"])
        categoricas = [x for x in categoricas if x not in ["id","Target"]]
    
        def nombre_(x):
          return "C"+str(x)
            
        cb = list(combinations(categoricas,2))
        p_value, modalidades, nombre_var = [], [], []
    
        base2 = df.get(categoricas).copy()
        for k in base2.columns:
            base2[k] = base2[k].map(nombre_)
    
        base2["Target"] = df["Target"].copy()
    
        for k in range(len(cb)):
        # Variable con interacción
            base2[cb[k][0]] = base2[cb[k][0]]
            base2[cb[k][1]] = base2[cb[k][1]]
    
            base2[cb[k][0]+"__"+cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]
    
        # Prueba chi cuadrado
            c1 = pd.DataFrame(pd.crosstab(base2["Target"],base2[cb[k][0]+"__"+cb[k][1]]))
            pv = stats.chi2_contingency(c1)[1]
    
        # Número de modalidades por categoría
            mod_ = len(base2[cb[k][0]+"__"+cb[k][1]].unique())
    
        # Guardar p value y modalidades
            nombre_var.append(cb[k][0]+"__"+cb[k][1])
            modalidades.append(mod_)
            p_value.append(pv)
    
        pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})
        pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=8),].sort_values(["p value"],ascending=True).head()
    
        def indicadora(x):
          if x==True:
            return 1
          else:
            return 0
        
        seleccion1 = list(pc.loc[(pc["p value"]<=0.20) & (pc["Num Modalidades"]<=8),"Variable"])
        sel1 = base2.get(seleccion1)
        
        contador = 0
        for k in sel1:
            if contador==0:
                lb1 = pd.get_dummies(sel1[k],drop_first=True)
                lb1.columns = [k + "_" + x for x in lb1.columns]
            else:
                lb2 = pd.get_dummies(sel1[k],drop_first=True)
                lb2.columns = [k + "_" + x for x in lb2.columns]
                lb1 = pd.concat([lb1,lb2],axis=1)
            contador = contador + 1
        
        for k in lb1.columns:
          lb1[k] = lb1[k].map(indicadora)
        
        lb1["Target"] = df["Target"].copy()
        lb1.head(3)
    
        cat_cuanti = [(x,y) for x in numericas for y in categoricas]
    
        v1, v2, pvalores_min, pvalores_max  = [], [], [], []
        
        for j in cat_cuanti:
            k1 = j[0]
            k2 = j[1]
        
            g1 = pd.get_dummies(df[k2])
            lt1 = list(g1.columns)
        
            for k in lt1:
                g1[k] = g1[k] * df[k1]
        
            g1["Target"] = df["Target"].copy()
        
            pvalues_c = []
            for y in lt1:
                mue1 = g1.loc[g1["Target"]==0,y].to_numpy()
                mue2 = g1.loc[g1["Target"]==1,y].to_numpy()
                mue3 = g1.loc[g1["Target"]==2,y].to_numpy()
        
                try:
                  pval = (stats.kruskal(mue1,mue2)[1]<=0.20)
                  if pval==True:
                      pval = 1
                  else:
                      pval = 0
                except ValueError:
                  pval = 0
                pvalues_c.append(pval)
        
            min_ = np.min(pvalues_c) # Se revisa si alguna de las categorías no es significativa
            max_ = np.max(pvalues_c) # Se revisa si alguna de las categorías es significativa
            v1.append(k1) # nombre de la variable 1
            v2.append(k2) # nombre de la variable 2
            
            pvalores_min.append(np.round(min_,2))
            pvalores_max.append(np.round(max_,2))
    
        pc2 = pd.DataFrame({'numerica':v1,'Categórica':v2,'p value':pvalores_min, 'p value max':pvalores_max})
        pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),]
    
        v1 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"numerica"])
        v2 = list(pc2.loc[(pc2["p value"]==1) & (pc2["p value max"]==1),"Categórica"])
    
        for j in range(len(v1)):
    
            if j==0:
                g1 = pd.get_dummies(df[v2[j]],drop_first=True)
                lt1 = list(g1.columns)
                for k in lt1:
                    g1[k] = g1[k] * df[v1[j]]
                g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
            else:
                g2 = pd.get_dummies(df[v2[j]],drop_first=True)
                lt1 = list(g2.columns)
                for k in lt1:
                    g2[k] = g2[k] * df[v1[j]]
                g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
                g1 = pd.concat([g1,g2],axis=1)
        
        g1["Target"] = df["Target"].copy()
        g1.head(5)
    
        var_cuad = list(pcuadrado1["Variable2"])
        base_modelo1 = base_cuadrado.get(var_cuad+["Target"])
        base_modelo1["Target"] = base_modelo1["Target"].map(int)
    
        cov = list(base_modelo1.columns)
        cov = [x for x in cov if x not in ["Target"]]
    
        X1 = base_modelo1.get(cov)
        y1 = base_modelo1.get(["Target"])
    
        modelo1 = XGBClassifier()
        modelo1 = modelo1.fit(X1,y1)
    
        importancias = modelo1.feature_importances_
        imp1 = pd.DataFrame({'Variable':X1.columns,'Importancia':importancias})
        imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
        imp1 = imp1.sort_values(["Importancia"],ascending=False)
        imp1.index = range(imp1.shape[0])
    
        var_int = list(pxy["Variable"])
        base_modelo2 = base_interacciones.get(var_int+["Target"])
        base_modelo2["Target"] = base_modelo2["Target"].map(int)
    
        cov = list(base_modelo2.columns)
        cov = [x for x in cov if x not in ["Target"]]
    
        X2 = base_modelo2.get(cov)
        y2 = base_modelo2.get(["Target"])
    
        modelo2 = XGBClassifier()
        modelo2 = modelo2.fit(X2,y2)
    
        importancias = modelo2.feature_importances_
        imp2 = pd.DataFrame({'Variable':X2.columns,'Importancia':importancias})
        imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
        imp2 = imp2.sort_values(["Importancia"],ascending=False)
        imp2.index = range(imp2.shape[0])
    
        var_raz = list(prazones["Variable"])
        base_modelo3 = base_razones1.get(var_raz+["Target"])
        base_modelo3["Target"] = base_modelo3["Target"].map(int)
    
        cov = list(base_modelo3.columns)
        cov = [x for x in cov if x not in ["Target"]]
        
        X3 = base_modelo3.get(cov)
        y3 = base_modelo3.get(["Target"])
        
        modelo3 = XGBClassifier()
        modelo3 = modelo3.fit(X3,y3)
        
        importancias = modelo3.feature_importances_
        imp3 = pd.DataFrame({'Variable':X3.columns,'Importancia':importancias})
        imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
        imp3 = imp3.sort_values(["Importancia"],ascending=False)
        imp3.index = range(imp3.shape[0])
    
        lb1["Target"] = lb1["Target"].map(int)
    
        cov = list(lb1.columns)
        cov = [x for x in cov if x not in ["Target"]]
        
        X4 = lb1.get(cov)
        y4 = lb1.get(["Target"])
        
        modelo4 = XGBClassifier()
        modelo4 = modelo4.fit(X4,y4)
        
        importancias = modelo4.feature_importances_
        imp4 = pd.DataFrame({'Variable':X4.columns,'Importancia':importancias})
        imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
        imp4 = imp4.sort_values(["Importancia"],ascending=False)
        imp4.index = range(imp4.shape[0])
    
        g1["Target"] = g1["Target"].map(int)
    
        cov = list(g1.columns)
        cov = [x for x in cov if x not in ["Target"]]
        
        X5 = g1.get(cov)
        y5 = g1.get(["Target"])
        
        modelo5 = XGBClassifier(objective="multi:softmax", num_class=3)
        modelo5 = modelo5.fit(X5,y5)
        
        importancias = modelo5.feature_importances_
        imp5 = pd.DataFrame({'Variable':X5.columns,'Importancia':importancias})
        imp5["Importancia"] = imp5["Importancia"] * 100 / np.sum(imp5["Importancia"])
        imp5 = imp5.sort_values(["Importancia"],ascending=False)
        imp5.index = range(imp5.shape[0])
    
        c2 = list(imp1.iloc[0:3,0]) # Variables al cuadrado
    
        razxy = list(imp3.iloc[0:3,0])
        cxy = list(imp2.iloc[0:3,0])
        catxy = list(imp4.iloc[0:3,0])
    
            # Variables cuantitativas (Activar D1)
        D1 = df.get(numericas).copy()
        
        # Variables categóricas
        D2 = df.get(categoricas).copy()
        for k in categoricas:
            D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()
        
        # Variables al cuadrado (Activar D1)
        cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
        cuadrado = [x[0] for x in cuadrado]
        
        for k in cuadrado:
            D1[k+"_2"] = D1[k] ** 2
        
        # Interacciones cuantitativas (Activar D1)
        result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]
        
        for k in result:
            D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        # Razones
        result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
        for k in result2:
            k2 = k[0]
            D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        # Interacciones categóricas
        result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
        for k in result3:
            D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
    
        base_modelo = pd.concat([D1,D4],axis=1)
        base_modelo["Target"] = df["Target"].copy()
        base_modelo["Target"] = base_modelo["Target"].map(int)
        base_modelo.head(3)
    
        formatos = pd.DataFrame(base_modelo.dtypes).reset_index()
        formatos.columns = ["Variable","Formato"]
        cuantitativas_bm = list(formatos.loc[formatos["Formato"]!="object",]["Variable"])
        categoricas_bm = list(formatos.loc[formatos["Formato"]=="object",]["Variable"])
        cuantitativas_bm = [x for x in cuantitativas_bm if x not in ["Target"]]
        categoricas_bm = [x for x in categoricas_bm if x not in ["Target"]]
        return cuantitativas_bm, categoricas_bm, base_modelo


    def model_noing1(self, categoricas, numericas, df):
        exp_clf101 = setup(data=df, target='Target', session_id = 123, train_size=0.7,
        numeric_features = numericas,
        categorical_features = categoricas)
    
        dt = create_model('lightgbm')
    
        hyperparameters = dt.get_params()
        
        
        results = pull()
        accuracy_mean_noing1 = results.iloc[-2]['Accuracy']  
    
        param_grid_bayesian = {
            'n_estimators': [50,100,200],
            'max_depth': [3,5,7],
            'min_child_samples': [50,150,200]
        }
        # Perform Bayesian Search
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
    
        predictions_test = predict_model(tuned_dt)
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train'))
        y_train = get_config('y_train')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        
        # Variables cuantitativas (Activar D1)
        D1 = prueba.get(numericas).copy()
        
        # Variables categóricas
        D2 = prueba.get(categoricas).copy()
        for k in categoricas:
          D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()
        
        # Variables al cuadrado (Activar D1)
        for k in cuadrado:
          D1[k+"_2"] = D1[k] ** 2
        
        # Interacciones cuantitativas (Activar D1)
        for k in result:
          D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        # Razones
        for k in result2:
          k2 = k[0]
          D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        # Interacciones categóricas
        for k in result3:
          D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
        base_modelo2 = prueba
    
        df_test3 = base_modelo2.copy()
    
        predictions3 = predict_model(final_dt, data=df_test3)
    
        y_hat_noing1 = final_dt.predict(df_test3)
        
        return accuracy_mean_noing1, y_hat_noing1
    
    def model_noing2(self, categoricas, numericas, df):
        exp_clf101 = setup(data=df, target='Target', session_id = 123, train_size=0.7,
        numeric_features = numericas,
        categorical_features = categoricas)
    
        dt = create_model('xgboost')
    
        hyperparameters = dt.get_params()
        
        
        results = pull()
        accuracy_mean_noing2 = results.iloc[-2]['Accuracy']  
    
        param_grid_bayesian = {
            'n_estimators': [50,100,200],
            'max_depth': [3,5,7],
            'min_child_samples': [50,150,200]
        }
        # Perform Bayesian Search
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
    
        predictions_test = predict_model(tuned_dt)
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train'))
        y_train = get_config('y_train')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        
        # Variables cuantitativas (Activar D1)
        D1 = prueba.get(numericas).copy()
        
        # Variables categóricas
        D2 = prueba.get(categoricas).copy()
        for k in categoricas:
          D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()
        
        # Variables al cuadrado (Activar D1)
        for k in cuadrado:
          D1[k+"_2"] = D1[k] ** 2
        
        # Interacciones cuantitativas (Activar D1)
        for k in result:
          D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        # Razones
        for k in result2:
          k2 = k[0]
          D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        # Interacciones categóricas
        for k in result3:
          D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
        base_modelo2 = prueba
    
        df_test3 = base_modelo2.copy()
    
        predictions3 = predict_model(final_dt, data=df_test3)
    
        y_hat_noing2 = final_dt.predict(df_test3)
        
        return accuracy_mean_noing2, y_hat_noing2
    
    def model_noing3(self, categoricas, numericas, df):
        # Configurar el entorno para clasificación
        exp_clf101 = setup(data=df, 
                           target='Target', 
                           session_id=123, 
                           train_size=0.7,
                           numeric_features=numericas,
                           categorical_features=categoricas)
    
        # Crear el modelo Gradient Boosting Classifier
        dt = create_model('gbc')
    
        # Obtener los hiperparámetros actuales del modelo
        hyperparameters = dt.get_params()
        
        # Obtener los resultados del modelo inicial
        results = pull()
        
        # Calcular la media de la exactitud (Accuracy)
        accuracy_mean_noing3 = results['Accuracy'].mean()
    
        # Definir el espacio de hiperparámetros para la búsqueda bayesiana
        param_grid_bayesian = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    
        # Ajustar el modelo usando búsqueda bayesiana
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, 
                              search_library='scikit-optimize', 
                              search_algorithm='bayesian', 
                              fold=5)
    
        # Realizar predicciones con el modelo ajustado
        predictions_test = predict_model(tuned_dt)
    
        # Obtener datos de entrenamiento y prueba
        X_train = get_config('X_train')
        y_train = get_config('y_train')
        X_test = get_config('X_test')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
    
        # Realizar predicciones sobre los datos de prueba
        predictions_test_final3 = predict_model(final_dt, data=X_test)
    
        # Retornar la media de la exactitud y las predicciones finales
        return accuracy_mean_noing3, predictions_test_final3

    def model_ing1(self, cuantitativas_bm, categoricas_bm, base_modelo):
        exp_clf101 = setup(data=base_modelo, target='Target', session_id = 123, train_size=0.7,
        numeric_features = cuantitativas_bm,
        categorical_features = categoricas_bm)
    
        dt = create_model('lightgbm')
    
        hyperparameters = dt.get_params()
        
        
        results = pull()
        accuracy_mean_ing1 = results.iloc[-2]['Accuracy']  
    
        param_grid_bayesian = {
            'n_estimators': [50,100,200],
            'max_depth': [3,5,7],
            'min_child_samples': [50,150,200]
        }
        # Perform Bayesian Search
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
    
        predictions_test = predict_model(tuned_dt)
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train'))
        y_train = get_config('y_train')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        
        # Variables cuantitativas (Activar D1)
        D1 = prueba.get(numericas).copy()
        
        # Variables categóricas
        D2 = prueba.get(categoricas).copy()
        for k in categoricas:
          D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()
        
        # Variables al cuadrado (Activar D1)
        for k in cuadrado:
          D1[k+"_2"] = D1[k] ** 2
        
        # Interacciones cuantitativas (Activar D1)
        for k in result:
          D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        # Razones
        for k in result2:
          k2 = k[0]
          D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        # Interacciones categóricas
        for k in result3:
          D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
        base_modelo2 = pd.concat([D1,D4],axis=1)
    
        df_test3 = base_modelo2.copy()
    
        predictions3 = predict_model(final_dt, data=df_test3)
    
        y_hat_ing1 = final_dt.predict(df_test3)
        
        return accuracy_mean_ing1, y_hat_ing1
    
    def model_ing2(self, cuantitativas_bm, categoricas_bm, base_modelo):
        exp_clf101 = setup(data=base_modelo, target='Target', session_id = 123, train_size=0.7,
        numeric_features = cuantitativas_bm,
        categorical_features = categoricas_bm)
    
        dt = create_model('xgboost')
    
        hyperparameters = dt.get_params()
        
        
        results = pull()
        accuracy_mean_ing2 = results.iloc[-2]['Accuracy']  
    
        param_grid_bayesian = {
            'n_estimators': [50,100,200],
            'max_depth': [3,5,7],
            'min_child_samples': [50,150,200]
        }
        # Perform Bayesian Search
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, search_library='scikit-optimize', search_algorithm='bayesian',fold=5)
    
        predictions_test = predict_model(tuned_dt)
        predictions_train = predict_model(tuned_dt, data=exp_clf101.get_config('X_train'))
        y_train = get_config('y_train')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
        
        # Variables cuantitativas (Activar D1)
        D1 = prueba.get(numericas).copy()
        
        # Variables categóricas
        D2 = prueba.get(categoricas).copy()
        for k in categoricas:
          D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()
        
        # Variables al cuadrado (Activar D1)
        for k in cuadrado:
          D1[k+"_2"] = D1[k] ** 2
        
        # Interacciones cuantitativas (Activar D1)
        for k in result:
          D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]
        
        # Razones
        for k in result2:
          k2 = k[0]
          D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]]+0.01)
        
        # Interacciones categóricas
        for k in result3:
          D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]
    
        base_modelo2 = pd.concat([D1,D4],axis=1)
    
        df_test3 = base_modelo2.copy()
    
        predictions3 = predict_model(final_dt, data=df_test3)
    
        y_hat_ing2 = final_dt.predict(df_test3)
        
        return accuracy_mean_ing2, y_hat_ing2
    
    def model_ing3(self, cuantitativas_bm, categoricas_bm, base_modelo):
        # Configurar el entorno para clasificación
        exp_clf101 = setup(data=base_modelo, 
                           target='Target', 
                           session_id=123, 
                           train_size=0.7,
                           numeric_features=cuantitativas_bm,
                           categorical_features=categoricas_bm)
    
        # Crear el modelo Gradient Boosting Classifier
        dt = create_model('gbc')
    
        # Obtener los hiperparámetros actuales del modelo
        hyperparameters = dt.get_params()
        
        # Obtener los resultados del modelo inicial
        results = pull()
        
        # Calcular la media de la exactitud (Accuracy)
        accuracy_mean_ing3 = results['Accuracy'].mean()
    
        # Definir el espacio de hiperparámetros para la búsqueda bayesiana
        param_grid_bayesian = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    
        # Ajustar el modelo usando búsqueda bayesiana
        tuned_dt = tune_model(dt, custom_grid=param_grid_bayesian, 
                              search_library='scikit-optimize', 
                              search_algorithm='bayesian', 
                              fold=5)
    
        # Realizar predicciones con el modelo ajustado
        predictions_test = predict_model(tuned_dt)
    
        # Obtener datos de entrenamiento y prueba
        X_train = get_config('X_train')
        y_train = get_config('y_train')
        X_test = get_config('X_test')
        y_test = get_config('y_test')
    
        # Finalizar el modelo
        final_dt = finalize_model(tuned_dt)
    
        # Realizar predicciones sobre los datos de prueba
        predictions_test_final3 = predict_model(final_dt, data=X_test)
    
        # Retornar la media de la exactitud y las predicciones finales
        return accuracy_mean_ing3, predictions_test_final3

    def evaluation_model(self, ing, num, cuantitativas_bm, categoricas_bm, base_modelo, categoricas, numericas, df):
        accuracy = None
        predictions = None

        if ing == "True":
            if num == 1:
                accuracy, predictions = self.model_ing1(cuantitativas_bm, categoricas_bm, base_modelo)
            elif num == 2:
                accuracy, predictions = self.model_ing2(cuantitativas_bm, categoricas_bm, base_modelo)
            elif num == 3:
                accuracy, predictions = self.model_ing3(cuantitativas_bm, categoricas_bm, base_modelo)
        else:
            if num == 1:
                accuracy, predictions = self.model_noing1(categoricas, numericas, df)
            elif num == 2:
                accuracy, predictions = self.model_noing2(categoricas, numericas, df)
            elif num == 3:
                accuracy, predictions = self.model_noing3(categoricas, numericas, df)

        return accuracy

    def ML_FLOW(self):
        try:
            categoricas, numericas, df = self.load_data()
            cuantitativas_bm, categoricas_bm, base_modelo = self.load_dataing()
            
            accuracy_noing1, _ = self.model_noing1(categoricas, numericas, df)
            accuracy_noing2, _ = self.model_noing2(categoricas, numericas, df)
            accuracy_noing3, _ = self.model_noing3(categoricas, numericas, df)
            
            accuracy_ing1, _ = self.model_ing1(cuantitativas_bm, categoricas_bm, base_modelo)
            accuracy_ing2, _ = self.model_ing2(cuantitativas_bm, categoricas_bm, base_modelo)
            accuracy_ing3, _ = self.model_ing3(cuantitativas_bm, categoricas_bm, base_modelo)
            
            # You need to define 'ing' and 'num' here based on your requirements
            ing = "True"  # or "False"
            num = 1  # or 2 or 3
            
            accuracy = self.evaluation_model(ing, num, cuantitativas_bm, categoricas_bm, base_modelo, categoricas, numericas, df)
            
            return {'success': True, 'accuracy': accuracy}
        except Exception as e:
            return {'success': False, 'message': str(e)}