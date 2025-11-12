from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model, ensemble, svm, model_selection, metrics
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from collections import Counter, defaultdict, deque



def aplicar_svr(lista_30, lista_15, lista_6, lista_4, lista_sig):
    X = np.column_stack((lista_30, lista_15, lista_6, lista_4))
    y = np.array(lista_sig)
    # Definir el modelo SVR con kernel RBF (muy usado para relaciones no lineales)
    svr_model = svm.SVR(kernel='rbf')
    # Definir una rejilla de hiperparámetros para ajustar el parámetro de penalización C y el gamma del kernel.
    p_grid = {
        "C": [ 118.509189,118.509191, 118.509192],
        "gamma": ['scale', 'auto', 0.0175003, 0.0175004]  
    }
    # Usar GridSearchCV para buscar los mejores hiperparámetros usando validación cruzada
    grid_search = model_selection.GridSearchCV(svr_model, p_grid, cv=4, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    # Extraemos el mejor modelo encontrado, la mejor puntuación, y los parámetros óptimos.
    best_svr = grid_search.best_estimator_
    cv_score = grid_search.best_score_
    best_params = grid_search.best_params_
    preds_all = best_svr.predict(X)
    errors_pct_all = (preds_all - y) / (y + 1e-15)
    mpe_all = np.mean(errors_pct_all)
    scores = cross_val_score(svr_model, X, y, cv=4, scoring='neg_mean_squared_error')  
    mean_score = scores.mean()    # promedio de los 5 folds
    std_score  = scores.std()     # desviación estándar, para ver variabilidad
    #print(f"CV : {mean_score:.5f}   Desv: {std_score:.5f}    CV  ==> {cv_score:.5f}")
    
    return best_svr, cv_score, mpe_all, errors_pct_all


def last_interval_expand(L30, L15, L6, L4, Lsig, uu, t):
    min_size=40
    if t==0:
        edges = [2.5, 2.65, 2.8, 2.9, 2.95, 3.0, 3.05, 3.1, 3.2, 3.35, 3.5]
    elif t==1:
        edges = [3.2, 3.5, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.3, 5.6]
    elif t==2:
        edges = [5, 5.3, 5.4, 5.5, 5.6, 5.7, 6]
    elif t==4:
        edges = [1.8, 1.85, 1.9, 1.95, 2.0, 2.1]
    elif t==5:
        edges = [4.0, 4.2, 4.4, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.3, 5.5, 5.7]
    elif t==6:
        edges = [2.6, 3.0, 3.2, 3.3, 3.4, 3.45, 3.5, 3.6, 3.7, 3.9, 4.1, 4.5]
    else:
        edges = [7, 7.4, 7.8, 8.2, 8.6, 9.0, 9.4, 9.8, 10.2, 10.6, 11.0, 11.3]

    # 1) Índice de bin (con recorte para incluir 6.0)
    bin_idx = np.digitize(uu, edges, right=False)
    bin_idx = max(0, min(bin_idx, len(edges)-1))

    # 2) Crear máscara para el bin principal
    arr15 = np.array(L6, dtype=float)
    if bin_idx == 0:
        left, right = -np.inf, edges[0]
    else:
        left, right = edges[bin_idx-1], edges[bin_idx]
    mask = (arr15 >= left) & (arr15 < right)

    # 3) Si es primer o último bin y faltan datos, sumar vecino
    if mask.sum() < min_size:
        if bin_idx == 0:
            # sumar el segundo tramo (3.0 ≤ x < 3.5)
            m2 = (arr15 >= edges[0]) & (arr15 < edges[1])
            mask = mask | m2
        elif bin_idx == len(edges)-1:
            # sumar el penúltimo tramo (5.5 ≤ x < 6.0)
            m2 = (arr15 >= edges[-2]) & (arr15 < edges[-1])
            mask = mask | m2

    # 4) Filtrar todas las listas
    L30a  = [v for v, m in zip(L30,  mask) if m]
    L15a  = [v for v, m in zip(L15,  mask) if m]
    L6a   = [v for v, m in zip(L6,   mask) if m]
    L4a   = [v for v, m in zip(L4,   mask) if m]
    Lsiga = [v for v, m in zip(Lsig, mask) if m]

    return L30a, L15a, L6a, L4a, Lsiga,  bin_idx, len(L4a) 


def last_interval(L30: List[float], L15: List[float], L6: List[float], L4: List[float], Lsig: List[float], uu: float, n_bins: int = 9) -> Tuple[List[float], List[float], List[float], List[float]]:
    # 1) Redondear min/max de L15 a múltiplos de .5
    arr15 = np.array(L15, dtype=float)
    min0 = math.floor(arr15.min()*2) / 2
    max0 = math.ceil (arr15.max()*2) / 2
    # 2) Media y diferencia
    mean = (min0 + max0) / 2
    d    = max0 - min0
    # 3) Bordes de los intervalos
    #    Creamos n_bins+1 bordes desde mean-0.5d hasta mean+0.5d
    edges = np.linspace(mean - 0.5*d, mean + 0.5*d, num=n_bins+1)
    # 4) Encontrar intervalo del último valor
    last_val = uu
    # digitize devuelve índice de borde derecho, con shift -1 conseguimos 0-based
    bin_idx  = np.digitize(last_val, edges, right=False) - 1
    # Cap en [0, n_bins-1]
    bin_idx = max(0, min(n_bins-1, bin_idx))
    # 5) Máscara booleana: mismo intervalo
    left_edge  = edges[bin_idx]
    right_edge = edges[bin_idx + 1]
    # Incluimos el límite izquierdo y excluimos el derecho salvo que sea el último bin
    if bin_idx == n_bins-1:
        mask = (arr15 >= left_edge) & (arr15 <= right_edge)
    else:
        mask = (arr15 >= left_edge) & (arr15 <  right_edge)

    # 6) Construir las listas filtradas
    L30a  = [v for v, m in zip(L30,  mask) if m]
    L15a  = [v for v, m in zip(L15,  mask) if m]
    L6a   = [v for v, m in zip(L6,   mask) if m]
    L4a   = [v for v, m in zip(L6,   mask) if m]
    Lsiga = [v for v, m in zip(Lsig, mask) if m]
        
    return L30a, L15a, L6a, L4a, Lsiga


def sliding_global(serie, k, dominio, last_n=20):
    """Retorna un diccionario {v: prob} con prob para cada valor en 'dominio'. """
    # 1) Lista de conteos de “siguiente” para todas las ventanas
    conteos = [serie[i : i + k].count(serie[i + k]) for i in range(len(serie) - k)]
    hist_counts    = Counter(conteos)
    total_ventanas = len(conteos)
    # 2) Tomamos solo las últimas last_n ventanas (o todas si hay menos)
    conteos_ult = conteos[-last_n:]
    # 3) Calculamos probabilidades p(c) para esos últimos conteos, Pero como queremos p(v) para cada valor, primero vemos
    # cuántas veces aparece v en la última ventana real
    p_global = [hist_counts[c] / total_ventanas for c in conteos]
    ultima_ventana = serie[-k:]
    #Pr, _, _, _, _, rz = procesar_e_imprimir_regresion(0, 4, conteos, 2, 0, 0, 7)
    print("Ultimos:", conteos[-5:])
    #yY=procesar_lista_tres(conteos, 1, 0)
    #print_colored_stats(yY, 0, Forma=0)

    #    y a partir de ahí buscamos p(c_v)
    probs = {}
    for v in dominio:
        c_v      = ultima_ventana.count(v)
        # hist_counts.get(c_v, 0) → cuántas ventanas históricas tuvieron ese conteo
        p_v      = hist_counts.get(c_v, 0) / total_ventanas
        probs[v] = round(p_v, 3)
    
    return   probs #, conteos, p_global[-last_n:],


def Dicc_probabilidad_ordenado(lista_numeros, I_ini=0, I_fin=10, cant=30,ventanas=(15, 20)):
    dicc1 = [sliding_global(lista_numeros, ventana, list(range(I_ini, I_fin))) 
    for ventana in ventanas]

    sums = defaultdict(float)
    for d in dicc1:
        for key, val in d.items():
            # si la clave es lista la convertimos a tupla
            hkey = tuple(key) if isinstance(key, list) else key
            sums[hkey] += val
    # 3) Calcular promedio
    n = len(dicc1)
    avg = {k: sums[k] / n for k in sums}

    return avg


def filtrar_segmentos_df(errores_30, errores_15, errores_6, errores_4, lista_sig, u, min_count=100):
    
    # 1) Crear DataFrame
    df = pd.DataFrame({
        "err30": errores_30,
        "err15": errores_15,
        "err6" : errores_6,
        "err4" : errores_4,
        "sig"  : lista_sig
    })
    
    # 2) Tolerancias sucesivas: 0%, 3%, 6%
    tolerancias = [0.00, 0.025, 0.5, 0.75, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4]
    df_filtrado = pd.DataFrame()  # placeholder
    for rtol in tolerancias:
        if rtol == 0.0:
            # ciclo exacto
            mask = df["err6"] == u
            etiqueta = "exacto"
        else:
            #ciclo con tolerancia rtol
            low, high = sorted([u*(1-rtol), u*(1+rtol)])
            mask = df["err6"].between(low, high)
            etiqueta = f"±{rtol*100:.0f}%"
        df_filtrado = df[mask]

        # si ya alcanzamos el mínimo, rompemos el bucle
        if len(df_filtrado) >= min_count:
            break

    # 5) Extraer las sublistas resultantes
    errores_30a = df_filtrado["err30"].tolist()
    errores_15a = df_filtrado["err15"].tolist()
    errores_6a  = df_filtrado["err6"] .tolist()
    errores_4a  = df_filtrado["err4"] .tolist()
    lista_siga  = df_filtrado["sig"] .tolist()

    return errores_30a, errores_15a, errores_6a, errores_4a, lista_siga, df_filtrado



def aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig):
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para regresión ponderada.")
        return None
    # Preparación de datos
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    X_ols = sm.add_constant(X)
    # Regresión OLS inicial
    modelo_ols = sm.OLS(y, X_ols).fit()
    residuos = modelo_ols.resid
    ### 🟢 MÉTODO 1: Pesos inversos al error (el que ya tenías)
    pesos_1 = 1 / (residuos ** 2 + 1e-6)
    modelo_wls_1 = sm.WLS(y, X_ols, weights=pesos_1).fit(cov_type='HC0')
    
    ### 🔵 MÉTODO 2: Pesos con raíz cuadrada del error
    pesos_2 = 1 / np.sqrt(residuos ** 2 + 1e-6)
    modelo_wls_2 = sm.WLS(y, X_ols, weights=pesos_2).fit(cov_type='HC0')
    
    # Extraer coeficientes para ambas versiones
    coeficientes_1 = modelo_wls_1.params
    coeficientes_2 = modelo_wls_2.params
    # Calcular métricas para ambas versiones
    resultados = {
        "Método 1: Pesos inversos al error": {
            "Intercepto": coeficientes_1[0],
            "Coef. lista_30": coeficientes_1[1],
            "Coef. lista_15": coeficientes_1[2],
            "Coef. lista_6": coeficientes_1[3],
            "Porcentaje de variabilidad explicada por el modelo (cuanto mayor, mejor)": modelo_wls_1.rsquared,
            "Promedio del error al cuadrado entre lo predicho y lo real (menores valores son mejores)": metrics.mean_squared_error(y, modelo_wls_1.predict(X_ols)),
            "Promedio absoluto de error entre lo predicho y lo real (indica desviación en las mismas unidades)": metrics.mean_absolute_error(y, modelo_wls_1.predict(X_ols))
        },
        "Método 2: Pesos con raíz cuadrada del error": {
            "Intercepto": coeficientes_2[0],
            "Coef. lista_30": coeficientes_2[1],
            "Coef. lista_15": coeficientes_2[2],
            "Coef. lista_6": coeficientes_2[3],
            "Porcentaje de variabilidad explicada por el modelo (cuanto mayor, mejor)": modelo_wls_2.rsquared,
            "Promedio del error al cuadrado entre lo predicho y lo real (menores valores son mejores)": metrics.mean_squared_error(y, modelo_wls_2.predict(X_ols)),
            "Promedio absoluto de error entre lo predicho y lo real (indica desviación en las mismas unidades)": metrics.mean_absolute_error(y, modelo_wls_2.predict(X_ols))
        }
    }
    return resultados


def aplicar_regresion_elasticnet(Yy, lista_30, lista_15, lista_6, lista_sig):
    #alpha=0.86, l1_ratio=0.14, max_iter=8000, tol=0.000001
    
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para Elastic Net.")
        return None
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    m_iter=6000
    tole=0.00001

    if Yy==0:
        modelo_enet = linear_model.ElasticNet(alpha=0.95, l1_ratio=0.0001, max_iter=m_iter, tol=tole).fit(X, y)
    elif Yy==1:
        modelo_enet = linear_model.ElasticNet(alpha=0.0035, l1_ratio=0.7, max_iter=m_iter, tol=tole).fit(X, y)
    elif Yy==2:
        modelo_enet = linear_model.ElasticNet(alpha=0.004, l1_ratio=0.5, max_iter=m_iter, tol=tole).fit(X, y)
    else:
        modelo_enet = linear_model.ElasticNet(alpha=0.0001, l1_ratio=0.75, max_iter=m_iter, tol=tole).fit(X, y)
    
    return modelo_enet.intercept_, modelo_enet.coef_


def aplicar_regresion_robusta(lista_30, lista_15, lista_6, lista_sig):
    if not all([lista_30, lista_15, lista_6, lista_sig]) or not len(lista_30) == len(lista_15) == len(lista_6) == len(lista_sig):
        print("Error en los datos para regresión robusta.")
        return None

    # Preparación de datos
    X = np.array([lista_30, lista_15, lista_6]).T
    y = np.array(lista_sig)
    X_ols = sm.add_constant(X)  # Agregar intercepto
    # Aplicar regresión robusta con método HuberT
    modelo_rlm = sm.RLM(y, X_ols, M=sm.robust.norms.HuberT()).fit()
    # Calcular un pseudo R² manualmente:
    ss_res = np.sum(modelo_rlm.resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_pseudo = 1 - ss_res / ss_tot if ss_tot != 0 else None
    coeficientes = modelo_rlm.params

    return {
        "Intercepto": coeficientes[0],
        "Coef. lista_30": coeficientes[1],
        "Coef. lista_15": coeficientes[2],
        "Coef. lista_6": coeficientes[3],
        "Estimación de variabilidad explicada por el modelo (Pseudo R²)": r2_pseudo
    }



def calcular_probabilidades_regresion(params_wls, params_enet, lista_30, lista_15, lista_6):
    if params_wls is None or params_enet is None or not lista_30 or not lista_15 or not lista_6:
        return {"WLS": None, "ElasticNet": None}
    if len(lista_30) != len(lista_15) or len(lista_30) != len(lista_6):
        print("Error: Las listas de promedios tienen longitudes inconsistentes.")
        return {"WLS": None, "ElasticNet": None}

    ultima_media_30 = lista_30[-1] if lista_30 else 0
    ultima_media_15 = lista_15[-1] if lista_15 else 0
    ultima_media_6 = lista_6[-1] if lista_6 else 0
    prediccion_wls = params_wls[0] + params_wls[1] * ultima_media_30 + params_wls[2] * ultima_media_15 + params_wls[3] * ultima_media_6
    intercepto_enet, coefs_enet = params_enet
    prediccion_enet = intercepto_enet + coefs_enet[0] * ultima_media_30 + coefs_enet[1] * ultima_media_15 + coefs_enet[2] * ultima_media_6

    return {"WLS": prediccion_wls, "ElasticNet": prediccion_enet}


def calcular_regresion_wls_metodo1(lista_30, lista_15, lista_6, lista_sig):
     
    params = aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Método 1: Pesos inversos al error"]["Intercepto"],
            "Coef_lista_30": params["Método 1: Pesos inversos al error"]["Coef. lista_30"],
            "Coef_lista_15": params["Método 1: Pesos inversos al error"]["Coef. lista_15"],
            "Coef_lista_6": params["Método 1: Pesos inversos al error"]["Coef. lista_6"]
        }
    else:
        return None


def calcular_regresion_wls_metodo2(lista_30, lista_15, lista_6, lista_sig):
      
    params = aplicar_regresion_ponderada(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Método 2: Pesos con raíz cuadrada del error"]["Intercepto"],
            "Coef_lista_30": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_30"],
            "Coef_lista_15": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_15"],
            "Coef_lista_6": params["Método 2: Pesos con raíz cuadrada del error"]["Coef. lista_6"]
        }
    else:
        return None


def calcular_regresion_robusta(lista_30, lista_15, lista_6, lista_sig):
        
    params = aplicar_regresion_robusta(lista_30, lista_15, lista_6, lista_sig)
    if params:
        return {
            "Intercepto": params["Intercepto"],
            "Coef_lista_30": params["Coef. lista_30"],
            "Coef_lista_15": params["Coef. lista_15"],
            "Coef_lista_6": params["Coef. lista_6"],
            "R2": params["Estimación de variabilidad explicada por el modelo (Pseudo R²)"]
        }
    else:
        return None


def predecir_con_regresion(parametros, params_enet, lista_30, lista_15, lista_6):
    # Primero, convierte el diccionario 'parametros' en una lista en el orden esperado:
    if parametros is None:
        return None
    parametros_lista = [
        parametros["Intercepto"],
        parametros["Coef_lista_30"],
        parametros["Coef_lista_15"],
        parametros["Coef_lista_6"]
    ]
    return calcular_probabilidades_regresion(parametros_lista, params_enet, lista_30, lista_15, lista_6)


def procesar_regresiones(datos, Xx):
    lista_30, lista_15, lista_6, lista_sig = datos
    # Aseguramos que se usan segmentos consistentes:
    datos_regresion = (
        lista_30[-len(lista_sig):],
        lista_15[-len(lista_sig):],
        lista_6[-len(lista_sig):],
        lista_sig
    )
    
    # Extraer parámetros para cada método
    params_wls1 = calcular_regresion_wls_metodo1(*datos_regresion)
    params_wls2 = calcular_regresion_wls_metodo2(*datos_regresion)
    params_rlm  = calcular_regresion_robusta(*datos_regresion)
    params_enet = aplicar_regresion_elasticnet(Xx, lista_30, lista_15, lista_6, lista_sig)  # Suponiendo que esta función ya existe
    # Calcular predicciones
    resultados_m1 = predecir_con_regresion(params_wls1, params_enet, lista_30, lista_15, lista_6)
    resultados_m2 = predecir_con_regresion(params_wls2, params_enet, lista_30, lista_15, lista_6)
    resultados_rlm = predecir_con_regresion(params_rlm, params_enet, lista_30, lista_15, lista_6)
    
    return {
        "WLS_Metodo1": resultados_m1,
        "WLS_Metodo2": resultados_m2,
        "RLM": resultados_rlm,
    }





def inferir_probabilidades_bayesianas(orden_digitos, historial_posiciones):
    evidence_history = historial_posiciones[-40:]
    prior_history = historial_posiciones[:-40]
    print()
    # Inicializamos conteos para cada posición (usaremos posiciones 1 a N, donde N = len(orden_digitos))
    num_posiciones = len(orden_digitos)  # normalmente 10
    prior_counts = {pos: 0 for pos in range(1, num_posiciones + 1)}
    evidence_counts = {pos: 0 for pos in range(1, num_posiciones + 1)}
    
    # Contar ocurrencias en el prior
    for pos in prior_history:
        if 1 <= pos <= num_posiciones:
            prior_counts[pos] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en el prior.")
    # Contar ocurrencias en la evidencia
    for pos in evidence_history:
        if 1 <= pos <= num_posiciones:
            evidence_counts[pos] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en la evidencia.")
            
    # La distribución posterior para la posición i es:
    # posterior(prob_i) = (prior_counts[i] + evidence_counts[i]) / (total_prior + total_evidence)
    total_prior = sum(prior_counts.values())
    total_evidence = sum(evidence_counts.values())  # debería ser 40 si todo está bien
    total_combined = total_prior + total_evidence
    
    posterior_probs = {}
    for pos in range(1, num_posiciones + 1):
        posterior_probs[pos] = (prior_counts[pos] + evidence_counts[pos]) / total_combined
    
    # Mostramos información de chequeo
    print("-- Prior counts (por posición) --")
    print(prior_counts)
    print("-- Evidence counts (últimas 40 jugadas) --")
    print(evidence_counts)
    print("-- Posterior (distribución de posiciones) --")
    print(posterior_probs)
    
    # Reconversión: asignamos la probabilidad calculada para cada posición
    # al dígito correspondiente segun el orden en orden_digitos.
    # Si la posición es 1 (1-indexado), corresponde a orden_digitos[0].
    final_probabilidades = {}
    for pos in range(1, num_posiciones + 1):
        digito = orden_digitos[pos - 1]
        final_probabilidades[digito] = posterior_probs[pos]
    return final_probabilidades


def inferir_probabilidades_bayesianas1(orden_digitos, historial_posiciones):
    num_posiciones = len(orden_digitos)  # Número total de posiciones
    print("Cantidad Posiciones",num_posiciones)
    # Calculamos los totales de prior y evidencia
    total_evidence = sum(orden_digitos.values())
    print("Total Prior",total_prior)
    total_prior = sum(historial_posiciones.values())
    print("Total Evidence",total_evidence)
    total_combined = total_prior + total_evidence

    # Calculamos probabilidades posteriores
    posterior_probs = {}
    for digito in orden_digitos:
        posterior_probs[digito] = (orden_digitos[digito] + historial_posiciones[digito]) / total_combined
        print(f"Dígito {digito} => orden: {orden_digitos[digito]}, historial: {historial_posiciones[digito]}")
    
    # Devolvemos el diccionario con las mismas claves (0, 1, 2, …)
    return posterior_probs



def inferir_probabilidades_bayesianas(frecuencias, alpha_prior):
    # Usamos un prior uniforme si no se proporciona
    if alpha_prior is None:
        alpha_prior = {i: 1 for i in range(10)}
    
    # Calculamos la suma total del prior (en este caso, 10, ya que 1 para cada dígito)
    total_alpha = sum(alpha_prior.values())
    # Total de observaciones (suma de todas las frecuencias)
    total_frecuencias = sum(frecuencias.get(i, 0) for i in range(10))
    # Suma total del posterior
    total_posterior = total_alpha + total_frecuencias
    # Calculamos la media del posterior para cada dígito:
    probabilidades_posterior = {}
    for i in range(10):
        # alpha posterior: prior + observación
        alpha_post = alpha_prior.get(i, 1) + frecuencias.get(i, 0)
        probabilidades_posterior[i] = alpha_post / total_posterior
    
    return probabilidades_posterior



def probabilidades_bayes(siguiente, x, y):
    if not siguiente:
        return {i: 0 for i in range(x, y)}
    frecuencia = {i: siguiente.count(i) for i in range(x, y)}
    total = len(siguiente)
    
    return {num: freq / total if total > 0 else 0 for num, freq in frecuencia.items()}
