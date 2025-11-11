import pandas as pd
import numpy as np
import pymc as pm
from arviz import summary, plot_trace
import statsmodels.api as sm
from collections import Counter, defaultdict, deque
from sklearn import linear_model, ensemble, svm, model_selection, metrics
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from typing import List, Dict, Tuple, Optional
import pdb
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime
import math
import re
import statistics
from statistics import mean
from collections import defaultdict

def probando(lista):
    resumen_200 = analizar_lista_pct(lista, pct=200, clip_vi = False)
    resumen_200 = analizar_lista_pct(lista, pct=180, clip_vi = False)
    resumen_200 = analizar_lista_pct(lista, pct=150, clip_vi = False)
    resumen_200 = analizar_lista_pct(lista, pct=100, clip_vi = False)


def construir_prefix_sums(arr: List[int]) -> List[int]:
    ps = [0] * (len(arr) + 1)
    for i, v in enumerate(arr):
        ps[i+1] = ps[i] + v
    return ps

def suma_intervalo(ps: List[int], a: int, b: int) -> int:
    return ps[b+1] - ps[a]

def analizar_lista_pct(
    lista: List[int],
    inicio_offset: int = 200,
    g_min: int = 3,
    g_max: int = 10,
    pct: float = 200.0,
    clip_min: float = 1.0,
    clip_max: float = 10.0,
    eps_denom: float = 1e-9,
    clip_vi: bool = True
) -> Dict[Tuple[int,int], Dict[str,float]]:
    
    n = len(lista)
    
    idx_start = n - inicio_offset
    ps = construir_prefix_sums(lista)

    errores_por_par: Dict[Tuple[int,int], List[float]] = defaultdict(list)
    errores_global: List[float] = []

    for g in range(g_max, g_min - 1, -1):
        M = max(1, int(round(g * (pct / 100.0))))
        k = 0
        while True:
            group_end = idx_start - 1 - k
            group_start = group_end - (g - 1)
            if group_start < 0:
                break

            # suma del grupo
            sum_group = suma_intervalo(ps, group_start, group_end)

            # avg_general: M elementos hacia atrás desde group_start (no incluido)
            to_idx = group_start - 1
            if to_idx < 0:
                k += 1
                continue
            from_idx = max(0, to_idx - (M - 1))
            window_len = to_idx - from_idx + 1
            if window_len <= 0:
                k += 1
                continue
            sum_window = suma_intervalo(ps, from_idx, to_idx)
            avg_general = sum_window / window_len

            # vi: raw then clipped
            vi_raw = avg_general * (g + 1) - sum_group
            if clip_vi:
                vi = max(clip_min, min(clip_max, vi_raw))
            else:
                vi = vi_raw
            #vi = max(clip_min, min(clip_max, vi_raw))

            # vr: valor inmediatamente anterior a group_start (to_idx)
            target_idx = to_idx
            
            if not (0 <= target_idx < n):
                k += 1
                continue
            
            vr = lista[target_idx]
            if clip_vi:
                if vr==0:
                    vr=1
            else:
                vr=vr+0.0001

            #denom = vr if abs(vr) > eps_denom else (eps_denom if vr >= 0 else -eps_denom)
            error_rel = (vi - vr) / vr

            errores_por_par[(g, M)].append(error_rel)
            errores_global.append(error_rel)

            k += 1

    resumen: Dict[Tuple[int,int], Dict[str,float]] = {}
    for par, vals in errores_por_par.items():
        cnt = len(vals)
        mean = (sum(vals) / cnt) if cnt else math.nan
        std = statistics.pstdev(vals) if cnt else math.nan
        resumen[par] = {'count': cnt, 'mean': mean, 'std': std}

    promedio_global_all = (sum(errores_global) / len(errores_global)) if errores_global else math.nan

    # Impresión resumida y top 10 (ordenado por mean)
    print(f"Promedio global de todos los errores relativos (vi - vr)/vr: {promedio_global_all:.3f}")
    pares_ordenados = sorted(
        ((par, info['mean'], info['std']) for par, info in resumen.items() if not math.isnan(info['mean'])),
        key=lambda x: x[1]
    )
    top_10 = pares_ordenados[:5]
    print("\nTop 10 duplas (g, M) con menor error medio (mean ± std):")
    for (g, M), avg, sd in top_10:
        print(f"  Grupo g = {g}, M = {M}  ->  mean = {avg:.3f}, std = {sd:.3f}, count = {resumen[(g,M)]['count']}")

    return resumen


def print_Histo(seq: List[int], k: int = 3) -> None:
    
    # 1) Posición de la última aparición de cada dígito
    last_pos = [-1] * 10
    # 2) Colas de tamaño fijo para capturar los gaps
    gaps = [deque(maxlen=k) for _ in range(10)]

    # 3) Bucle único: calcular gap y actualizar última posición
    for i, v in enumerate(seq):
        lp = last_pos[v]
        if lp >= 0:
            gaps[v].append(i - lp)
        last_pos[v] = i

    # 4) Impresión de resultados
    print("")
    for d in range(10):
        # convierte deque a lista al imprimir
        print(f"{d}: {list(gaps[d])}")
    print("")


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


def Zonas_Series(lista, Preg, *c):
    #zonificar = lambda x: 1 if x < 4 else 2 if x < 7 else 3
    zonificar = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4
    zonas = lista.map(zonificar).astype(int)

    if c:
        primer_impres(zonas, 0, 1, 2, 3, c[0])
        
    else:
        primer_impres(zonas, 0, 1, 2, 3)
        
    #ab=fun_promedios(zonas, 0, 0, 2)
    Pr2, _, _, _, _, rz  = procesar_e_imprimir_regresion("Zonas Numeros", Preg, 4, zonas, 2, 4, 1, 5)

    return Pr2



def Zonas_Numeros(lista):
    print("\t\tZonas de numeros trabajando....")
    zonificar = lambda x: 1 if x < 4 else 2 if x < 7 else 3
    # Aplicamos la función a cada elemento de la lista original
    zonas = [zonificar(x) for x in lista]
    arr = np.array(zonas)
    
    print("")
    #ab=fun_promedios(zonas, 0, 0, 2)
    Pr2, _, _, _, _, rz = procesar_e_imprimir_regresion("Zonas Numeros", 0, 4, zonas, 2, 4, 1, 4)
    print("\x1b[38;5;55m--    --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --\x1b[0m")
    print("")
    
    return Pr2


def zones(serie: pd.Series, window: int = 50, max_count: int = None) -> list:
    
    if max_count is None:
        max_count = window + 1

    zones = []
    n = len(serie)
    for i in range(window, n):
        current = int(serie.iat[i])
        # 1) Ventana previa de tamaño `window`
        prev_window = serie.iloc[i-window : i].tolist()
        # 2) Conteo “jugadas sin caer” por dígito
        counts = {}
        rev = prev_window[::-1]
        for d in range(10):
            try:
                # idx_rev = 0 si cayó en la jugada justo anterior
                idx_rev = rev.index(d)
                counts[d] = idx_rev + 1
            except ValueError:
                counts[d] = max_count
        # 3) Ordenar dígitos por conteo
        ordered = sorted(counts, key=counts.get)
        # 4) Posición del dígito actual y clasificación
        pos = ordered.index(current)
        if counts[current] < 4:
            zone = 1
        elif counts[current] < 8:
            zone = 2
        elif counts[current] < 12:
            zone = 3 
        else:
            zone = 4
        # Fuerza zona 3 si nunca apareció en la ventana
        if counts[current] == max_count:
            zone = 4
        zones.append(zone)

    max_count = 50
    last_window = serie.iloc[-window:].tolist()
    counts = {}
    rev = last_window[::-1] # Revertir la ventana para encontrar la aparición más reciente
    for d in range(10):
        try:
            idx_rev = rev.index(d)
            counts[d] = idx_rev + 1
        except ValueError:
            counts[d] = max_count

    ordered = sorted(counts, key=counts.get)
    print("Zonas por Jugadas....")
    
    print("\033[95m" + "\t".join(map(str, ordered))+ "\033[0m")        
    #Pr1, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Caidas", 0, zones, 2, 4, 1, 5)
    Pr2, _, _, _, _, rz = procesar_e_imprimir_regresion("Posicion Caidas", 0, 4, zones, 2, 4, 1, 5)
    #Pr3, _, _, _, _ = procesar_e_imprimir_regresion("Posicion Caidas", 5, zones, 2, 4, 1, 5)
    #y=Sumar_listas(Pr1, Pr2, Pr3)
    print(" ".join(f"{v:.3f} " for v in Pr2))
    return zones


def zones_by_freq(serie: List[int], Preg, *cae) -> List[int]:
    zonifica = lambda x: 1 if x < 4 else 2 if x < 6 else 3 if x < 8 else 4
    zonas = [zonifica(x) for x in serie]
    
    N10 = zonas[-75:]
    zona=pd.Series(N10)
    if cae:
        c=zonifica(cae[0])
        primer_impres(zona, 0, 1, 2, 3, c)
    else:
        primer_impres(zona, 0, 1, 2, 3)
    
    ab=fun_promedios(zonas, 0, 0, 2)
    Pr2, _, _, _, _, rz = procesar_e_imprimir_regresion("Zona Frecuencias", Preg, 4, zonas, 2, 4, 1, 5)
    
    return Pr2


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
    Pr, _, _, _, _, rz = procesar_e_imprimir_regresion("-- Posicion segun Frec en 20 --", 0, 4, conteos, 2, 0, 0, 7)
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



def jerarquia_histo(lista: List[int], start: int = 30) -> List[int]:
    n = len(lista)
    # Convertimos start a índice 0-based
    i0 = start - 1
    # Diccionario con la última ocurrencia (0-based) de cada dígito
    last_occ = {d: -1 for d in range(10)}
    resultado: List[int] = []
    
    for i, val in enumerate(lista):
        last_occ[val] = i
        # Solo a partir de i0 y sin procesar el último elemento
        if i >= i0 and i < n - 1:
            # Construir orden de caída:
            # 1) Dígitos presentes, ordenados por última aparición descendente
            presentes = [d for d in range(10) if last_occ[d] != -1]
            presentes.sort(key=lambda d: -last_occ[d])
            # 2) Dígitos ausentes, ordenados ascendentemente
            ausentes = [d for d in range(10) if last_occ[d] == -1]
            ausentes.sort()
            orden_caida = presentes + ausentes
            # Buscar la posición 1-based del siguiente valor
            siguiente = lista[i + 1]
            posicion = orden_caida.index(siguiente) + 1
            resultado.append(posicion)
            
    return resultado


def porcentaje_coincidencias(F_d: dict, datos: list) -> dict:
    n = len(datos)
    limite = int(n * 0.95)        # 90% de n, redondeado hacia abajo
    primeros = datos[:limite]
    ultimos  = datos[limite:]
    c1, c2 = Counter(primeros), Counter(ultimos)
    n1, n2 = len(primeros), len(ultimos)

    errores = {}
    for k, valor in F_d.items():
        freq1 = c1[valor] / n1 if n1 else 0
        freq2 = c2[valor] / n2 if n2 else 0
        errores[k] = abs(freq1 - freq2)/freq1 if freq1 else 0
    return errores


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


def prediccion_bayesiana(lista_30, lista_15, lista_6, lista_sig):
    # Convertir listas a arrays
    np_lista_30 = np.array(lista_30)
    np_lista_15 = np.array(lista_15)
    np_lista_6  = np.array(lista_6)
    np_lista_sig = np.array(lista_sig)
    
    # Preparar datos de entrenamiento (todos menos el último)
    X_train = np.column_stack((np_lista_30[:-1], np_lista_15[:-1], np_lista_6[:-1]))
    y_train = np_lista_sig[:-1]
    
    # x_new: el último registro para predecir
    x_new = np.array([np_lista_30[-1], np_lista_15[-1], np_lista_6[-1]])
    
    # Definir el modelo bayesiano
    with pm.Model() as modelo:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = alpha + pm.math.dot(X_train, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
        trace = pm.sample(2000, tune=2000, chains=4, target_accept=0.98, return_inferencedata=True)
    
    # Usar sample_posterior_predictive y obtener un diccionario simple
    with modelo:
        ppc = pm.sample_posterior_predictive(
            trace, 
            var_names=["alpha", "beta"], 
            random_seed=42, 
            return_inferencedata=False
        )
    
    # Extraer muestras usando el diccionario
    alpha_samples = ppc["alpha"]
    beta_samples = ppc["beta"]
    # Calcular predicciones para cada muestra
    predicciones = alpha_samples + np.dot(beta_samples, x_new)
    # Calcular la predicción final y el intervalo del 95%
    prediccion_media = np.mean(predicciones)
    pred_int2_5 = np.percentile(predicciones, 1)
    pred_int97_5 = np.percentile(predicciones, 99)
    
    return {
        "prediccion_media": prediccion_media,
        "int_95": (pred_int2_5, pred_int97_5),
        "trace": trace
    }


def leer_datos_excel(file_path):
    df = pd.read_excel(file_path)
    columna = pd.to_numeric(df['A'], errors='coerce').dropna()
    return columna


def obtener_siguiente_numero(columna):
    ultima_caida = columna.iloc[-1]
    return [columna[i + 1] for i in range(len(columna) - 1) if columna[i] == ultima_caida]


def Siguientes_lista(lista):
    ultima = lista[-1]
    return [siguiente for actual, siguiente in zip(lista, lista[1:]) 
            if actual == ultima]


def obtener_historial_caidas(columnas, maxi):
    caidas_columna = []
    ultimas_posiciones = [-1] * 10
    for i, idx in enumerate(columnas):
        valor=int(idx)
        if ultimas_posiciones[valor] == -1:
            jugadas = i + 1
        else:
            jugadas = i - ultimas_posiciones[valor]
        if jugadas > maxi:
            jugadas = maxi if jugadas % 2 == 0 else (maxi - 1)
        caidas_columna.append(jugadas)
        ultimas_posiciones[valor] = i
    return caidas_columna


def Semanas(columna):
    grupo = columna.tail(24)
     # Calcular cuántas jugadas han pasado desde la última aparición de cada número
    apariciones = {}
    for num in range(10):
        if num in grupo.tolist():
            # Encontrar la posición de la última aparición y calcular la distancia desde el final
            ultima_posicion = len(grupo) - 1 - grupo[::-1].tolist().index(num)
            distancia = len(grupo)  - ultima_posicion
        else:
            # Si el número no aparece en el grupo, asignar 40 como default
            if num % 2 == 0:
                distancia = 24
            else :
                distancia = 23
        apariciones[num] = distancia
    return apariciones


def ultima_jerarquia(columna):
    grupo = columna.tail(50)
    frecuencias = {num: grupo.tolist().count(num) for num in range(10)}  # Usa .count() en la lista
    #print(frecuencias)  # Ahora imprimirá las frecuencias correctamente
    return frecuencias


def ultima_jerarquia_Lista(columna):
    grupo = columna[-50:]  # Extrae los últimos 50 elementos de la lista
    frecuencias = {num: grupo.count(num) for num in range(10)}  # Cuenta ocurrencias
    #print(frecuencias)  # Muestra las frecuencias en consola
    return frecuencias


def calcular_jerarquias(columna):
    jerarquia = []
    posiciones = []
    for i in range(len(columna) - 2, 51,-1 ):
        grupo = columna[max(0, i - 49):i + 1]
        frecuencias = {num: grupo.value_counts().get(num, 0) for num in range(10)}
        primeras_apariciones = {num: (len(grupo) - 1 - grupo[::-1].tolist().index(num) if num in grupo.tolist() else 60) for num in range(10)}
        ordenados = sorted(frecuencias.items(), key=lambda x: (x[1], primeras_apariciones[x[0]]))
        jerarquia.append(ordenados)
        if i < len(columna) - 1:
            dato_sig = columna[i + 1]
            
            for pos, (num, _) in enumerate(ordenados):
                if num == dato_sig:
                    posiciones.append(pos + 1)
                    break
    jerarquia.reverse()
    posiciones.reverse()
    return jerarquia, posiciones


def calcular_alpha_prior(columna):
    limite = int(len(columna) * 0.9)  # Define el 90% del total
    grupo = columna.iloc[:limite]  # Obtiene la parte inicial de la columna
    alpha_prior = {num: grupo.tolist().count(num) for num in range(10)}  # Cuenta ocurrencias
    return alpha_prior


def calcular_alpha_prior_Lista(columna):
    limite = int(len(columna) * 0.9)  # Define el 90% del total
    grupo = columna[:limite]  # Extrae los últimos 50 elementos de la lista
    frecuencias = {num: grupo.count(num) for num in range(10)}  # Cuenta ocurrencias
    return frecuencias


def calcular_mayores_pares(columna):
    mayores = [num for num in columna if num > 4]
    pares = [num for num in columna if num % 2 == 0]
    return mayores, pares


def aplicar_regresion_logistica_mayor_menor(columna):
    if len(columna) < 70:
        print("Error: Insuficientes datos para regresión logística (mayor/menor).")
        return None
    X = np.array(columna[:-1]).reshape(-1, 1)
    y = np.array([1 if num > 4 else 0 for num in columna[1:]])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = linear_model.LogisticRegression().fit(X_train, y_train)
    ultimo_numero = np.array(columna.iloc[-1]).reshape(1, 1)
    return modelo.predict_proba(ultimo_numero)[0][1]


def aplicar_regresion_logistica_par_impar(columna):
    if len(columna) < 70:
        print("Error: Insuficientes datos para regresión logística (par/impar).")
        return None
    X = np.array(columna[:-1]).reshape(-1, 1)
    y = np.array([1 if num % 2 == 0 else 0 for num in columna[1:]])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = linear_model.LogisticRegression().fit(X_train, y_train)
    ultimo_numero = np.array(columna.iloc[-1]).reshape(1, 1)
    return modelo.predict_proba(ultimo_numero)[0][1]


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


def probabilidades_bayes(siguiente, x, y):
    if not siguiente:
        return {i: 0 for i in range(x, y)}
    frecuencia = {i: siguiente.count(i) for i in range(x, y)}
    total = len(siguiente)
    
    return {num: freq / total if total > 0 else 0 for num, freq in frecuencia.items()}


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


def ultimos_promedios_list(data: List[float], Va, xx) -> List[float]: 
    return [(Va + v) / xx for v in data.values()]


def ultimos_promedios_series(data: pd.Series) -> List[float]:
    if len(data) < 15:
        # no hay ni un solo promedio completo
        return pd.Series(dtype=float)
    medias = data.rolling(15).mean().dropna()
    return medias.iloc[-10:].tolist()


def calcular_promedios_de_errores(columna):
    PTotal = np.mean(columna)
    # Calcular lista_30 y errores_30
    lista_30 = [np.mean(columna[i - 30:i]) for i in range(80, len(columna))]
    errores_30 = [(p - PTotal) / PTotal for p in lista_30]
    # Calcular lista_15 y errores_15 con índice correcto
    lista_10 = [np.mean(columna[i - 15:i]) for i in range(80, len(columna))]
    errores_10 = [(p - PTotal) / PTotal for p in lista_10]
    # Calcular lista_4 y errores_4
    lista_4 = [np.mean(columna[i - 12:i]) for i in range(80, len(columna))]
    errores_4 = [(p - PTotal) / PTotal for p in lista_4]
    # Calcular lista_sig
    lista_sig = [np.mean(columna[i - 16:i + 1]) for i in range(80, len(columna) + 1)]
    lista_sig.pop()
    Tot14=sum(columna[-16:]) 
    u30=sum(columna[-30:])/len(columna[-30:])
    u10=sum(columna[-15:])/len(columna[-15:])
    u4=sum(columna[-12:])/len(columna[-12:])
    
    u30=(u30-PTotal)/PTotal
    u10=(u10-PTotal)/PTotal
    u4=(u4-PTotal)/PTotal

    return errores_30, errores_10, errores_4, lista_sig, Tot14, PTotal, u30, u10, u4


def calcular_promedios_y_errores(columna, Pos, tip):
    PTotal = np.mean(columna)
    if Pos<3:
        print("Datos muy altos") 
        # Calcular lista_30 y errores_30
        lista_30 = [np.mean(columna[i - 28:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 20:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_6 = [np.mean(columna[i - 10:i]) for i in range(40, len(columna))]
        errores_6 = [(p - PTotal) / PTotal for p in lista_6]
        lista_4 = [np.mean(columna[i - 3:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 11:i + 1]) for i in range(40, len(columna) + 1)]
        
        lista_sig.pop()
        Tot14=sum(columna[-11:])
        u30=sum(columna[-28:])/len(columna[-28:])
        u10=sum(columna[-20:])/len(columna[-20:])
        u6=sum(columna[-10:])/len(columna[-10:])
        u4=sum(columna[-3:])/len(columna[-3:])
        xxx=u6
        L_3, L_1, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, xxx, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (224, 112, 10)))

        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u6=(u6-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal

    elif Pos<5:
        # Calcular lista_30 y errores_30
        print("Datos Bajos")
        lista_30 = [np.mean(columna[i - 30:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 15:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_6 = [np.mean(columna[i - 12:i]) for i in range(40, len(columna))]
        errores_6 = [(p - PTotal) / PTotal for p in lista_6]
        lista_4 = [np.mean(columna[i - 3:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 16:i + 1]) for i in range(40, len(columna) + 1)]
        lista_sig.pop()    
        Tot14=sum(columna[-16:])
        u30=sum(columna[-30:])/len(columna[-30:])
        u10=sum(columna[-15:])/len(columna[-15:])
        u6=sum(columna[-12:])/len(columna[-12:])
        u4=sum(columna[-3:])/len(columna[-3:])
        xxx=u6
        L_3, L_1, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, xxx, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (224, 112, 10)))

        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u6=(u6-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal

    else:
        # Calcular lista_30 y errores_30
        print("Datos muy Bajos")
        lista_30 = [np.mean(columna[i - 35:i]) for i in range(40, len(columna))]
        errores_30 = [(p - PTotal) / PTotal for p in lista_30]
        # Calcular lista_15 y errores_15 con índice correcto
        lista_10 = [np.mean(columna[i - 25:i]) for i in range(40, len(columna))]
        errores_10 = [(p - PTotal) / PTotal for p in lista_10]
        # Calcular lista_4 y errores_4
        lista_6 = [np.mean(columna[i - 15:i]) for i in range(40, len(columna))]
        errores_6 = [(p - PTotal) / PTotal for p in lista_6]
        lista_4 = [np.mean(columna[i - 4:i]) for i in range(40, len(columna))]
        errores_4 = [(p - PTotal) / PTotal for p in lista_4]
        # Calcular lista_sig
        lista_sig = [np.mean(columna[i - 23:i + 1]) for i in range(40, len(columna) + 1)]
        lista_sig.pop()    
        Tot14=sum(columna[-23:])
        u30=sum(columna[-35:])/len(columna[-35:])
        u10=sum(columna[-25:])/len(columna[-25:])
        u6=sum(columna[-15:])/len(columna[-15:])
        u4=sum(columna[-4:])/len(columna[-4:])
        xxx=u6
        L_3, L_1, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, xxx, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_1, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}   {bn}",(225, 225, 225), (224, 112, 10)))

        u30=(u30-PTotal)/PTotal
        u10=(u10-PTotal)/PTotal
        u6=(u6-PTotal)/PTotal
        u4=(u4-PTotal)/PTotal
    
    print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}  {tam} \t\t Ultimos datos de 6:{u30:.3f}  {u10:.3f}   {u6:.3f}  {u4:.3f}",(225, 225, 225), (50, 250, 50)))
    return errores_30, errores_10, errores_6, errores_4, lista_sig, Tot14, PTotal, u30, u10, u6, u4, xxx, Pprom 


def procesar_e_imprimir_regresion(titulo,Pregunta, Pos, Lista, Nn, Tipo, start=0,stop=10):
    # Códigos ANSI
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[33;46m"
    RESET        = "\033[0m"
    
    print(aviso_ansi(f"Resultados para {titulo}:",(118, 5, 30), (240, 220, 90)))
    L_30, L_15, L_6, L_4, L_sig, Sum14, PROM, u3, u1, u6, u4, ud, p = calcular_promedios_y_errores(Lista, Pos, Tipo)
    
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_4, L_sig)
    nuevo_dato = np.array([[u3, u1, u6, u4]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    
    Pprom4=0
    l3, l1, l6, l4, ls=last_interval(L_30, L_15, L_6, L_4, L_sig, u1)
    if len(l3) > 60:
        best_svr4, cv_score4, PromT4, ETo4 = aplicar_svr(l3, l1, l6, l4, ls)
        #nuevo_dato4 = np.array([[u3, u1, u6, u4]])
        prediccion4 = best_svr4.predict(nuevo_dato)
        Pprom4=prediccion4[0]
    
    #probando(ETo)
    ab=fun_promedios(ETo, 2, 1, 3)
    #print("\nParámetros para SVR:", best_params)
    print(f"\033[31mGeneral :\033[0m", end="\t")
    print(f"\033[1;31;47m{PROM:.3f}\033[0m", end="\t\t") 
    
    # Verificar que todas las listas contengan datos, tengan la misma longitud y mayor a 50.
    if (L_30 and L_15 and L_6 and L_sig and len(L_30) == len(L_15) == len(L_6) == len(L_sig) and len(L_30) > 60):
        
        P_gral=Pprom
        if Pos<3:
            print(f"\x1b[48;5;202mRegresion:{Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        elif Pos<5:
            print(f"\x1b[48;5;157mRegresion:{Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        else:
            print(f"\x1b[43;5;223mRegresion:{Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        #print(f"'ElasticNet': {elasticnet1:.3f}   Otro: {Pprom4:.3f}")
        print(f"Otro: {Pprom4:.3f}")
        print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
        n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral, start, stop)
        
        minY=min(n_valores)
        maxY=max(n_valores)
        P_gral=P_gral*(1-ab)

        if P_gral<minY:  
            P_gral=minY*1.005
            
        if P_gral>maxY:
            P_gral=maxY*0.995
            
        n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral, start, stop, col_val="\033[92m", col_err="\033[92m")
        Valores=n_valores
        zz=(P_gral-minY)/(maxY-minY)
        print(f"\tNuevo Prom: {P_gral:.3f}\t\tRazon: {zz:.2f}")
        print("-----------------------          ------------------------------")
        Prom_Gral=P_gral

        if Pregunta==1:
            Prom_Gral = solicitar_nuevo_prom(minY, maxY)
            nuevos_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, Prom_Gral, start, stop, col_val="\033[92m", col_err="\033[92m")
            Valores=nuevos_valores
        
        return  errores_ajustados, Valores, Sum14, Prom_Gral, xx, zz

    else:
        print(f"No hay datos suficientes o las listas no cumplen las condiciones para {titulo}.")
        return None, None, None


def imprimir_valores_y_errores(s14, Po, p_gral, start=0, stop=10, col_val="\033[93m", col_err="\033[96m"):
    formateados = []
    errores     = []
    nuevos      = []
    ajustados   = []
    escalas     = []
    u           = []
    x=0
    
    for i in range(start, stop):
        if Po<3:
            val   = (s14 + i) / 12
            x=12
        elif Po<5:
            val   = (s14 + i) / 17
            x=17
        else:
            val   = (s14 + i) / 25
            x=25
        val=float(val)    
        err   = (val - p_gral) / p_gral
        escala=error_to_scale(err)
        formateados.append(f"{val:.3f}\t")
        errores.append(f"{err:.3f}")
        escalas.append(f"{escala:.1f}")
        ajustados.append(err * -0.999 if err < 0 else err)
        nuevos.append(val)
        
    for v in escalas:  
        u.append(str(v))  
    
    linea5 = "\t".join(u)
    print(f"{col_val}{' '.join(formateados)}\033[0m")
    print("\t".join(colorear2(float(e)) for e in errores))
    print(f"\033[95m{linea5}\033[0m")

    return nuevos, ajustados, x


def error_to_scale(err, lo=-0.10, hi=0.10, steps=10):
    # lo..hi dividido en 10 pasos produce 11 puntos; step size:
    step = (hi - lo) / steps
    # normalizamos y convertimos a escala 1..11 (puede dar decimales)
    value = 1 + (err - lo) / step
    # opcional: recortar fuera de rango
    return max(1.0, min(1.0 + steps, value))


def print_recencia(rec: Dict[int, int]) -> None:
    keys_l = sorted(rec.keys())
    vals_l = [rec[k] for k in keys_l]

    sorted_items = sorted(rec.items(), key=lambda kv: kv[1])
    keys_r, vals_r = zip(*sorted_items)

    # 2) Formatea con ancho fijo de columna
    col_w = 4
    sep_left = " │"
    left_idx_line   = "".join(f"{k:>{col_w}}" for k in keys_l) + sep_left
    left_vals_line  = "".join(f"{v:>{col_w}}" for v in vals_l) + sep_left
    right_idx_line  = "".join(f"{k:>{col_w}}" for k in keys_r)
    right_vals_line = "".join(f"{v:>{col_w}}" for v in vals_r)

    # 3) Genera border dinámico
    border_l = "-" * len(left_idx_line)
    border_r = "-" * len(right_idx_line)

    # 4) Imprime las 5 líneas: border / índices / border / valores / border
    spacer = "   "
    for left, right in [
        (border_l,      border_r),
        (left_idx_line, right_idx_line),
        (border_l,      border_r),
        (left_vals_line, right_vals_line),
        (border_l,      border_r),
    ]:
        print(f"{left}{spacer}{right}")






def crear_bordes(rango_min=-0.15, rango_max=0.15, ancho_bin=0.001):
    bordes = np.arange(rango_min, rango_max + ancho_bin, ancho_bin)
    # Primer y último bin absorben todo fuera de rango
    return np.concatenate(([-np.inf], bordes, [np.inf]))


def analizar_nuevos_errores(errores, edges, pct_totales, pct_recientes):
    n_dec=3
    # 1. Calcular diff con signo para cada error
    diffs = []
    for err in errores:
        idx = np.digitize(err, edges) - 1
        idx = np.clip(idx, 0, len(pct_totales) - 1)
        diffs.append(pct_totales[idx] - pct_recientes[idx])

    # 2. Convertir a strings con signo y ancho uniforme
    errores_str = [f"{e:+.{n_dec}f}" for e in errores]
    diffs_str   = [f"{d:+.{n_dec}f}" for d in diffs]
    ancho = max(max(len(s) for s in errores_str),
                max(len(s) for s in diffs_str)) + 2

    # 3. Construir separadores y líneas
    total_cols = len(errores_str)
    line_len   = total_cols * ancho + (total_cols - 1)
    sep        = "-" * line_len

    # 4. Imprimir
    print(sep)
    print(" ".join(s.center(ancho) for s in errores_str))
    print(sep)
    print(" ".join(s.center(ancho) for s in diffs_str))
    print(sep)



def Caidas_por_rango(valores, error):
    ultimos_n=30 
    rango_min=-0.15 
    rango_max=0.15 
    ancho_bin=0.001

    # 1. Calcular bordes
    edges = crear_bordes(rango_min, rango_max, ancho_bin)

    # 2. Histograma completo
    conteos_totales, _ = np.histogram(valores, bins=edges)
    pct_totales = conteos_totales / conteos_totales.sum()

    # 3. Histograma de los últimos N
    reciente = valores[-ultimos_n:] if len(valores) >= ultimos_n else valores
    conteos_recientes, _ = np.histogram(reciente, bins=edges)
    pct_recientes = conteos_recientes / conteos_recientes.sum()

    # 4. Diferencia absoluta de porcentajes
    diff_pct = np.abs(pct_totales - pct_recientes)

    # 5. Empaquetar resultados por bin
    intervals = list(zip(edges[:-1], edges[1:]))
    resultados = [
        {
            "intervalo": f"{start:.3f}…{end:.3f}",
            "pct_total": float(pct_totales[i]),
            "pct_reciente": float(pct_recientes[i]),
            "diff_pct": float(diff_pct[i])
        }
        for i, (start, end) in enumerate(intervals)
    ]

    analizar_nuevos_errores(error, edges, pct_totales, pct_recientes)

    return resultados, edges, pct_totales, pct_recientes


def Imprimir_datos_listas(stats: dict, mode: int = 0):
    # ANSI settings
    BG_GRAY = "\x1b[48;2;220;220;220m"
    RESET   = "\x1b[0m"
    # Variables que deben ir sin decimales
    int_vars = {"N", "Tpos", "Tigu", "Tneg","Sig", "Ant"}

    def fmt_scalar(name: str, val: float) -> str:
        """ Construye 'name=value' con fondo gris y texto rojo/azul o gris si es None.
        Enteros sin decimales, floats con 3 decimales.
        """
        # 1) Caso None
        if val is None:
            r, g, b = (128, 128, 128)   # gris para valores no disponibles
            txt = "N/A"
        else:
            # 2) decide color según signo
            r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
            if name in int_vars:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.3f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {name}={txt} "
            f"{RESET}"
        )
    
    def fmt_number(val: float) -> str:
        """ Sólo número (3 decimales), con fondo gris y texto rojo/azul. """
        r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
        txt = f"{val:.2f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {txt} "
            f"{RESET}"
        )

    # 1) Línea de escalares
    order = ["Ult", "N", "Tpos", "Tigu", "Tneg",
             "Ptot", "Ppos", "Pigu", "Pneg"]
    line1 = "  ".join(
        fmt_scalar(name, stats[name])
        for name in order
        if name in stats and not isinstance(stats[name], list)
    )
    print(line1)
    print("")
    # si sólo se pide la primera línea, cortamos
    if mode == 1:
        return

    # 2) Línea Siguiente (sig)
    sigs = stats.get("Sig", [])
    if sigs:
        y1=sum(sigs[-15:])/len(sigs[-15:]) if len(sigs[-15:])>0 else 0
        y2=sum(sigs[-10:])/len(sigs[-10:]) if len(sigs[-10:])>0 else 0
        y3=sum(sigs[-10:-5])/len(sigs[-10:-5]) if len(sigs[-10:])>0 else 0
        y4=sum(sigs[-5:])/len(sigs[-5:]) if len(sigs[-5:])>0 else 0
        
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")

    # 3) Línea Anterior (ant)
    ants = stats.get("Ant", [])
    if len(ants)>0:
        y1=sum(ants[-15:])/len(ants[-15:])
        y2=sum(ants[-10:])/len(ants[-10:])
        y3=sum(ants[-10:-5])/len(ants[-10:-5])
        y4=sum(ants[-5:])/len(ants[-5:])
    
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")
        print("")


def print_colored_stats(stats: dict, mode: int = 0, Forma=1):
    # ANSI settings
    BG_GRAY = "\x1b[48;2;220;220;220m"
    RESET   = "\x1b[0m"
    # Variables que deben ir sin decimales
    int_vars = {"N", "Tpos", "Tneg"}

    def fmt_scalar(name: str, val: float) -> str:
        """ Construye 'name=value' con fondo gris y texto rojo/azul o gris si es None.
        Enteros sin decimales, floats con 3 decimales.
        """
        # 1) Caso None
        if val is None:
            r, g, b = (128, 128, 128)   # gris para valores no disponibles
            txt = "N/A"
        else:
            # 2) decide color según signo
            r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
            if name in int_vars:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.3f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {name}={txt} "
            f"{RESET}"
        )

    def fmt_number(val: float) -> str:
        """ Sólo número (3 decimales), con fondo gris y texto rojo/azul. """
        r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
        if Forma==0:
            txt = f"{val:.2f}"
        elif Forma==1:
            txt = f"{val:.3f}"
        else:
            txt = f"{val:.0f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {txt} "
            f"{RESET}"
        )

    # 1) Línea de escalares
    order = ["Ult", "N", "Tpos", "Tneg",
             "Ptot", "Ppos", "Pneg"]
    line1 = "  ".join(
        fmt_scalar(name, stats[name])
        for name in order
        if name in stats and not isinstance(stats[name], list)
    )
    print(line1)
    print("")
    # si sólo se pide la primera línea, cortamos
    if mode == 1:
        return

    # 2) Línea Siguiente (sig)
    sigs = stats.get("Sig", [])
    if sigs:
        if len(sigs[-15:])!=0:
            y1=sum(sigs[-15:])/len(sigs[-15:])  
        else:
            y1=0
        
        if len(sigs[-10:])!=0:    
            y2=sum(sigs[-10:])/len(sigs[-10:]) 
        else:
            y2=0
        
        if len(sigs[-10:-5])!=0:
            y3=sum(sigs[-10:-5])/len(sigs[-10:-5]) 
        else:
            y3=0
        if len(sigs[-5:])!=0:
            y4=sum(sigs[-5:])/len(sigs[-5:])
        else:
            y4=0
    
        line2_vals = " ".join(fmt_number(v) for v in sigs[-10:])
        print(f"Sig. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")

    # 3) Línea Anterior (ant)
    ants = stats.get("Ant", [])
    if len(ants)>0:
        if len(ants[-15:])!=0:
            y1=sum(ants[-15:])/len(ants[-15:])  
        else:
            y1=0
        
        if len(ants[-10:])!=0:    
            y2=sum(ants[-10:])/len(ants[-10:]) 
        else:
            y2=0
        
        if len(ants[-10:-5])!=0:
            y3=sum(ants[-10:-5])/len(ants[-10:-5]) 
        else:
            y3=0
        if len(ants[-5:])!=0:
            y4=sum(ants[-5:])/len(ants[-5:])
        else:
            y4=0
    
        line2_vals = " ".join(fmt_number(v) for v in ants[-10:])
        print(f"Act. :\t{line2_vals}\t\tP15: {y1:.2f}   P10: {y2:.2f}   PM: {y3:.2f}   P5: {y4:.2f}")
        print("")


def aviso_ansi(texto: str, fg: tuple = (64, 34, 28), bg: tuple = (253, 226, 228)) -> str:
    fg_code = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m"
    bg_code = f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m"
    reset   = "\033[0m"
    return f"{fg_code}{bg_code}{texto}{reset}"


def solicitar_nuevo_prom(min_val, max_val):
    prompt  = aviso_ansi(
        f"⚠️  Introduce nuevo Prom_Gral "
        f"(entre {min_val:.3f} y {max_val:.3f}): ",(151,79,68),(191,183,182)
    )
    while True:
        try:
            nuevo = float(input(prompt))
        except ValueError:
            print(aviso_ansi("→ Entrada no válida. Sólo números."))
            continue

        if min_val <= nuevo <= max_val:
            return nuevo
        print(aviso_ansi(
            f"→ Fuera de rango. Debe ser entre {min_val:.3f} y {max_val:.3f}."
        ))


def colorear(valor, etiqueta, dec, t=0):
    # amarillo brillante para la etiqueta
    azul  = "\033[34m"
    rojo  = "\033[31m"
    amarillo = "\033[92m"
    reset = "\033[0m"

    num_color = azul if valor >= 0 else rojo
    if dec==0:
        return f"{amarillo}{etiqueta}{reset} = {num_color}{valor:.0f}{reset}"
    else:
        if t==0:
            return f"{amarillo}{etiqueta}{reset} = {num_color}{valor:.3f}{reset}"
        else:
            return f"{amarillo}{etiqueta}{reset} = {num_color}{valor:.4f}{reset}"
        
def colorear2(valor):
    fondo = "\033[104m"  # Fondo gris claro
    if valor >= 0:
        texto = "\033[34m" if valor > 0 else "\033[30m"  # Azul si >0, negro si ==0
    else:
        texto = "\033[31m"  # Rojo si negativo
    return f"{fondo}{texto}{valor:.3f}\033[0m"


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


def imprimir_tabla(Titulo, data, es_decimal=False, highlight_key=None):
    # ANSI colors
    RED = "\033[1;31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    # Caso diccionario: se mantienen las claves en el orden de inserción
    if isinstance(data, dict):
        claves = list(data.keys())
        cabecera = [str(k) for k in claves]
        valores = [data[k] for k in claves]
    # Caso lista: se usan los índices de la lista como cabecera
    elif isinstance(data, list):
        cabecera = [str(i) for i in range(len(data))]
        valores = data
    else:
        print("Tipo de dato no soportado (se esperaba diccionario o lista).")
        return

    # Formateo de la fila de datos:
    if es_decimal:
        # Se formatean los números a 4 dígitos decimales
        fila_datos = [f"{v:.3f}" for v in valores]
    else:
        fila_datos = [str(v) for v in valores]

    # Determina el ancho mínimo que se necesita para cada celda (según la mayor longitud entre cabecera y datos)
    ancho_min = max(max(len(s) for s in cabecera), max(len(s) for s in fila_datos))
    # Se añade un pequeño padding (2 espacios adicionales)
    ancho_celda = ancho_min + 2
    num_cols = len(cabecera)
    
    # índices de las 4 columnas centrales
    mid         = num_cols // 2
    offset      = 1 if num_cols % 2 else 0
    centro_idxs = list(range(mid - 2 + offset, mid + 2))

    # índice de la clave a resaltar en rojo
    idx_hl = None
    if highlight_key is not None:
        try:
            idx_hl = cabecera.index(str(highlight_key))
        except ValueError:
            pass

    # función de formateo de cada celda
    def fmt(s, i):
        cell = f"{s:>{ancho_celda}}"
        if i == idx_hl:
            return f"{RED}{cell}{RESET}"
        if i in centro_idxs:
            return f"{YELLOW}{cell}{RESET}"
        return cell

    # línea de borde
    borde = "-" * ((ancho_celda + 1) * num_cols + 1)

    # --- 4) Impresión de la tabla ---
    print(f"\n******  {Titulo}  *****")
    print(borde)
    print(" ".join(fmt(c, i) for i, c in enumerate(cabecera)) + " │")
    print(borde)
    print(" ".join(fmt(d, i) for i, d in enumerate(fila_datos)) + " │")
    print(borde)




def imprimir_tabla_N(titulo: str, data, es_decimal: bool = False, color_titulo: str = "default", blink: bool = False,
    light: bool = False):
    # ——— Configuración ANSI de colores/blink ———
    base = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
        "default": 39
    }
    bright = light or color_titulo.startswith("light_")
    clave = color_titulo.replace("light_", "") if bright else color_titulo
    codigo_color = base.get(clave.lower(), base["default"])
    if bright:
        codigo_color += 60
    codigos = []
    if blink:
        codigos.append("5")
    codigos.append(str(codigo_color))
    seq_inicio = f"\033[{';'.join(codigos)}m"
    seq_fin    = "\033[0m"

    # ——— Extraer cabecera y valores ———
    if isinstance(data, dict):
        claves = list(data.keys())
        cabecera = [str(k) for k in claves]
        valores  = [data[k] for k in claves]
    elif isinstance(data, list):
        cabecera = [str(i) for i in range(len(data))]
        valores  = data
    else:
        print("Tipo no soportado (esperado dict o list).")
        return

    # ——— Formatear valores ———
    fila_datos = []
    for v in valores:
        if es_decimal and isinstance(v, float):
            fila_datos.append(f"{v:.4f}")
        else:
            fila_datos.append(str(v))

    # ——— Calcular anchos por columna (contenido vs. cabecera) + padding ———
    n = len(cabecera)
    anchos = []
    for i in range(n):
        ancho_max = max(len(cabecera[i]), len(fila_datos[i]))
        anchos.append(ancho_max + 2)   # +2 espacios de padding

    # ——— Construir líneas de borde ———
    # Ej: +----+------+----+
    partes = ["+" + "-" * a for a in anchos]
    borde = "".join(partes) + "+"

    # ——— Construir filas de texto ———
    # Cabecera: | key0 | key1 | ...
    cab = "|"
    datos = "|"
    for i in range(n):
        cab += f" {cabecera[i].center(anchos[i]-2)} |"
        datos += f" {fila_datos[i].center(anchos[i]-2)} |"

    # ——— Impresión final ———
    print()
    # título centrado sobre la tabla
    ancho_tabla = len(borde)
    print(seq_inicio + titulo.center(ancho_tabla) + seq_fin)
    print(borde)
    print(cab)
    print(borde)
    print(datos)
    print(borde)


def calcular_probabilidades_desde_historial(orden_digitos, historial_posiciones):
    # 2. Inicializamos un diccionario para contar apariciones, para cada dígito
    conteos = {digito: 0 for digito in orden_digitos}
    
    # 3. Recorrer el historial.
    #    Se asume que los números del historial son posiciones 1-indexadas.
    for pos in historial_posiciones:
        index = pos - 1  # Convertir a índice 0-indexado
        if 0 <= index < len(orden_digitos):
            digito = orden_digitos[index]
            conteos[digito] += 1
        else:
            print(f"Advertencia: posición {pos} fuera de rango en el historial.")
       
    # 4. Normalizamos los conteos para obtener probabilidades.
    total = sum(conteos.values())
    if total > 0:
        probabilidades = {digito: conteos[digito] / total for digito in conteos}
    else:
        # Si no hay registros en el historial, puede hacerse una distribución uniforme u otra política
        probabilidades = {digito: 0.005 for digito in conteos}
    return probabilidades


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


def ordenar_por_valor(d, ascendente=True):
    if d is None:
        # No hay nada que ordenar; devuelvo lista vacía
        return []
        
    return dict(
        sorted(d.items(), key=lambda par: par[1], reverse=not ascendente)
    )


def mostrar_dict(d):
    for clave, valor in d.items():
        print(f"{clave}: {valor}")


def mostrar_formato(num):
    entero = int(num)
    decimal = round(num - entero, 4)
    dec_str = f"{decimal:.4f}"[2:]  # '1456', sin '0.'
    
    if num < 1:
        print(f".{dec_str}")
    else:
        print(f"{entero}.{dec_str}")


def Lista2_con_map(lista):
    """Aplica y = x//2 + 1 a cada elemento usando map+lambda."""
    return list(map(lambda x: x // 2 + 1, lista))

def Lista2_series(s: pd.Series) -> pd.Series:
    """Aplica y = x//2 + 1 a cada elemento de la Series y devuelve otra Series."""
    return s.map(lambda x: x // 2 + 1)

def Histo2_con_map(lista, T):
    """Aplica y = (x-1)//2 + 1 a cada elemento usando map+lambda."""
    return list(map(lambda x: (x-1) // T + 1, lista))


def ordenar_lista(lista: list, ascendente: bool = True) -> list:
    return sorted(lista, reverse=not ascendente)


def split_segments(H: List[int], n: int = 3) -> List[List[int]]:
    """Divide H en n trozos lo más parejos posible."""
    L = len(H)
    size = L // n
    segments = []
    for i in range(n-1):
        segments.append(H[i*size : (i+1)*size])
    segments.append(H[(n-1)*size : ])
    return segments


def compute_percentages(seg: List[int], possible: List[int]) -> Dict[int, float]:
    """ Cuenta cuántas veces aparece cada valor en 'possible' dentro de 'seg' y devuelve porcentaje (0–1).  """
    cnt = Counter(seg)
    total = len(seg) if seg else 1
    return {v: cnt.get(v, 0)/total for v in possible}


def mean_percentages(per_list: List[Dict[int, float]]) -> Dict[int, float]:
    """Dado un listado de dicts {v: pct}, devuelve su promedio por clave."""
    keys = per_list[0].keys()
    n = len(per_list)
    return {k: sum(d[k] for d in per_list)/n for k in keys}


def compute_fd_errors(
    mean_p: Dict[int,float],
    last_p: Dict[int,float],
    F_d:    Dict[int,int]
) -> Tuple[Dict[int,float], Dict[int,float]]:
    """ Devuelve dos dicts: error_abs_fd[k] = abs(last_p[x] - mean_p[x])
      error_rel_fd[k] = (last_p[x] - mean_p[x]) / mean_p[x] donde x = F_d[k]. """
    error_abs_fd = {}
    error_rel_fd = {}

    for k, x in F_d.items():
        # Si x no está en mean_p o last_p, saltamos o le ponemos 0
        m = mean_p.get(x)
        l = last_p.get(x)
        if m is None or l is None or m == 0:
            error_abs_fd[k] = 0
            error_rel_fd[k] = 0
        else:
            abs_err = abs(l - m)
            rel_err = (l - m) / m
            error_abs_fd[k] = abs_err
            error_rel_fd[k] = rel_err
    return error_abs_fd, error_rel_fd


def analyze_frecuencias(
    H: List[int],
    F_d: Dict[int,int],
    max_val: int,
    n_segments: int,
    last_n: int 
) -> Tuple[Dict[int,float], Dict[int,float], Dict[int,float], Dict[int,float]]:
    
    possible = list(range(1, max_val+1))
    # 1) Segmentar y calcular porcentajes históricos
    segs       = split_segments(H, n_segments)
    historical = [compute_percentages(s, possible) for s in segs]
    mean_p     = mean_percentages(historical)
    # 2) Porcentaje últimos N
    last_p     = compute_percentages(H[-last_n:], possible)
    # 3) Errores absolutos por valor
    error_abs_fd, error_rel_fd = compute_fd_errors(mean_p, last_p, F_d)
    # 4) Mapear errores según F_d
    error_fd = {k: error_abs_fd.get(k, 0.0) for k in F_d}

    return mean_p, last_p, error_abs_fd, error_rel_fd, error_fd

def LLama_Sigui(Colu):
    Sig_numeros = obtener_siguiente_numero(Colu)
    Ss=pd.Series(Sig_numeros)
    SHis=obtener_historial_caidas(Ss)
    F_dsig=Semanas(Ss)
    print(len(SHis))
    if len(SHis)> 100:
    #    Zonas_Histos(Sig_numeros, 0)
        c=procesar_Histogramas("Histograma de Siguientes", 2, 2, 30, Ss, F_dsig)
    return SHis


def Sumar_diccionarios(*dicts):
    if dicts and isinstance(dicts[-1], int):
        Tipo, dicts = dicts[-1], dicts[:-1]
    else:
        Tipo, dicts = 0, dicts

    total = defaultdict(float)
    cuenta = defaultdict(int)
    # Sumar y contar apariciones
    if Tipo==0:
        for d in dicts:
            for clave, valor in d.items():
                total[clave] += valor
                cuenta[clave] += 1
    else:
            for d in dicts:
                for clave, valor in d.items():
                    total[clave] += abs(valor)
                    cuenta[clave] += 1
    
    # Calcular promedio por clave
    return {clave: total[clave] / cuenta[clave] for clave in total}
    #return ordenar_por_valor(diccionario_sumado, ascendente=False)


def remapear_por_posicion(claves_ordenadas: list, dic_posiciones: dict)-> dict:
    resultado = {}
    n = len(claves_ordenadas)
    for pos, val in dic_posiciones.items():
        i = pos - 1
        print(f"  probando pos={pos} → i={i}, rango 0–{len(claves_ordenadas)-1}")
        if 0 <= i < n:
            resultado[claves_ordenadas[i]] = val
        else:
            pass
    return resultado


def promedios_y_errores_lista(data, zz, Pos, yy, tip, nx):
    n = len(data)
    pgral = sum(data) / n
    Predm=0
    def medias_ventana(k):
            return [ sum(data[i-k: i]) / k for i in range(60, n) ]

    if zz < 1:
        print("Datos muy altos")
        lista_30 = medias_ventana(45)
        lista_10 = medias_ventana(30)
        lista_6  = medias_ventana(20)
        lista_4  = medias_ventana(6)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(60, n+1) ]
        u30=sum(data[-44:])/len(data[-44:])
        u10=sum(data[-29:])/len(data[-29:])
        u6=sum(data[-19:])/len(data[-19:])
        u4=sum(data[-5:])/len(data[-5:])
        lista_sig.pop()
        #print(u6, tip)
        L_3, L_10, L_6, L_4, L_sig, bn, tam= last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, u6, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_10, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}",(225, 225, 225), (117, 174, 90)))
        
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u6=(u6-pgral)/pgral
        u4=(u4-pgral)/pgral
    
    elif  zz < 5:
        print("Datos Bajos")

        lista_30 = medias_ventana(25)
        lista_10 = medias_ventana(12)
        lista_6  = medias_ventana(10)
        lista_4  = medias_ventana(4)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(60, n+1) ]
        u30=sum(data[-24:])/len(data[-24:])
        u10=sum(data[-11:])/len(data[-11:])
        u6=sum(data[-9:])/len(data[-9:])
        u4=sum(data[-3:])/len(data[-3:])
        lista_sig.pop()
        #print(u6, tip)
        L_3, L_10, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, u6, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_10, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}",(225, 225, 225), (117, 174, 90)))
        
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u6=(u6-pgral)/pgral
        u4=(u4-pgral)/pgral
    else :
        print("Datos muy Bajos")
        #def medias_ventana(k):
        #    return [ sum(data[i-k: i]) / k for i in range(30, n) ]
        lista_30 = medias_ventana(35)
        lista_10 = medias_ventana(20)
        lista_6  = medias_ventana(15)
        lista_4  = medias_ventana(3)

        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(60, n+1) ]
        u30=sum(data[-34:])/len(data[-34:])
        u10=sum(data[-19:])/len(data[-19:])
        u6=sum(data[-14:])/len(data[-14:])
        u4=sum(data[-2:])/len(data[-2:])
        lista_sig.pop()
        L_3, L_10, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, u6, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_10, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}",(225, 225, 225), (117, 174, 90)))
        
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u6=(u6-pgral)/pgral
        u4=(u4-pgral)/pgral

    def errores(lista_medias):
        return [ (m - pgral) / pgral for m in lista_medias ]

    errores_30 = errores(lista_30)
    errores_15 = errores(lista_10)
    errores_6  = errores(lista_6)
    errores_4  = errores(lista_4)
                      
    # 4) suma de los últimos 14 valores
    suma_14 = sum(data[-(nx-1):])
    
    return errores_30, errores_15, errores_6, errores_4, lista_sig, suma_14, pgral, u30, u10, u6, u4, Predm, tam

def calcular_nuevos_y_errores(Valores: Dict[str, float], Sum14: float, Prom_Gral: float, Nx) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    errores_clave   = {}
    nuevos_clave    = {}
    formateados1    = []
    formateados2    = []
    Escalar         = []
    escalaprint     = []

    for clave, incremento in Valores.items():
        nuevo_valor = (Sum14 + incremento) / Nx
        error = (nuevo_valor - Prom_Gral) / Prom_Gral
        escala=error_to_scale(error)
        nuevos_clave[clave]  = nuevo_valor
        errores_clave[clave] = error
        Escalar.append(escala)
        formateados1.append(f"{nuevo_valor:.2f}")
        formateados2.append(f"{error:.3f}")
        escalaprint.append(f"{escala:.1f}")
    return nuevos_clave, errores_clave, formateados1, formateados2, escalaprint


def procesar_regresion_Histo(titulo, Pr, zz, P, Lista, Valores, Nn, tip, start=1, stop=15 ):
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[37;45m"
    RESET        = "\033[0m"
    
    print(aviso_ansi(f"Resultados para {titulo}:", (118, 5, 30), (240, 220, 90)))
    y = dict(sorted(Valores.items(), key=lambda item: item[1]))
    Predm=0
    
    L_30, L_15, L_6, L_4, L_sig, Sum14, PROM, u3, u1, u6, u4, Predm, tam = promedios_y_errores_lista(Lista, zz, P, y, tip, Nn)
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_4, L_sig)
    nuevo_dato = np.array([[u3, u1, u6, u4]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    print(u6)
    er_30a, er_15a, er_6a, er_4a, lis_siga, df_debug = filtrar_segmentos_df(L_30, L_15, L_6, L_4, L_sig, u6)
    
    if len(df_debug)>30:
        best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, er_4a, lis_siga)
        pred = best_svr1.predict(nuevo_dato)
        Predm= pred[0]

    #resumen_200 = analizar_lista_pct(ETo, pct=200)
    ab=fun_promedios(ETo, 2, 2, 3)
    #print("Parámetros para SVR:", best_params)
    print("\033[31mPromedio :\033[0m", end=" ")
    print(f"\033[1;91;47m{PROM:.3f}\033[0m", end="\t")  
    
    if len(L_30) < 70:
        print(f"No hay datos suficientes en {titulo}.")
        # Ruta por defecto (cuando no hay datos suficientes)
        default = ({i: 0 for i in range(1, 10)}, None, 0)
        return default
    
    if (L_30 and L_15 and L_6 and L_sig and 
        len(L_30) == len(L_15) == len(L_6) == len(L_sig) ):
        datos_regresion = (L_30, L_15, L_6, L_sig)
        
        P_gral=Pprom
        if zz <3:
            print(f"\x1b[48;5;202mRegresion: {Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        elif zz <5:
            print(f"\x1b[48;5;157mRegresion: {Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        else:
            print(f"\x1b[43;5;223mRegresion: {Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
        
        #print(f"'ElasticNet': {elasticnet1:.3f}\tF1: {Predm:.3f}")
        print(f"F1: {Predm:.3f} {tam}")
        print(" ")
        nuevos, errores, textoN, textoE, tt = calcular_nuevos_y_errores(y, Sum14, Pprom, Nn)
        c=[]
        i=[]
        
        for k, v in y.items():  
            c.append(str(k))
            i.append(str(v))   
        
        linea1 = "\t".join(c)
        linea2 = "\t".join(i)

        #print(f"\033[93m{linea1}\033[0m")
        print(f"\033[96m{linea2}\033[0m")
        linea5 = "\t".join(tt)
        linea3 = "\t".join(textoN)
        linea4 = "\t".join(textoE)
        print(f"\033[91m{linea3}\033[0m")
        print(f"\033[97m{linea4}\033[0m")
        print(f"\033[95m{linea5}\033[0m")
        minY = min(nuevos.values())
        maxY = max(nuevos.values())

        Pprom=Pprom*(1-ab)        
        if Pprom<minY:
            Pprom=minY*1.004
        if Pprom>maxY:
            Pprom=maxY*0.996
        
        nuevos, errores, texto,textoE, tt = calcular_nuevos_y_errores(y, Sum14, Pprom, Nn)
        #linea3 = "\t".join(textoN)
        
        #linea4 = "\t".join(textoE)
        #print(f"\033[91m{linea3}\033[0m")
        #print(f"\033[97m{linea4}\033[0m")
        print("Nuevo Prom: ", "{:.1f}".format(Pprom))
        print("--------       -----------       ---------        --------       ---------        --------")
        if Pr==1 :
            Prom_Gral = solicitar_nuevo_prom(minY, maxY)
            nuevos, errores, texto, textoE, tt = calcular_nuevos_y_errores(y, Sum14, Prom_Gral, Nn)
            linea = "\t".join(textoN)
            linea1 = "\t".join(textoE)
            linea2 = "\t".join(tt)
            print(f"\033[92m{linea}\033[0m")
            print(f"\033[95m{linea1}\033[0m")
            print(f"\033[95m{linea2}\033[0m")
        return  errores, nuevos, Sum14

    else:
        print(f"Las listas no cumplen las condiciones para {titulo}.")
        return None, None, None


def escalar_dic(d, escalar):
    return {k: v * escalar for k, v in d.items()}


def procesar_Histogramas(titulo: str, Preg, Zz:int, Pos: int, Nn, h_data, f_data: dict, tip, *proc_args):
    Y=len(h_data)
    default = ({i: 0 for i in range(1, 10)}, None, None)
        
    if Y > 120:
        Error_val, Nuevo_valor, Sum14 = procesar_regresion_Histo(titulo, Preg, Zz, Pos, h_data, f_data, Nn, tip, *proc_args)
        # 2) Ordenar frecuencias y errores
        if Sum14 !=0:
            caidas_ordenadas = ordenar_por_valor(f_data, ascendente=True)
            prom_ordenados   = ordenar_por_valor(Error_val, ascendente=True)
            

            llave_min = min(prom_ordenados, key=lambda k: abs(prom_ordenados[k])) #min(prom_ordenados, key=prom_ordenados.get)
            #valor_min = Error_val[llave_min]
            # 4) Imprimir tablas
            #imprimir_tabla("  Caídas", caidas_ordenadas, es_decimal=False, highlight_key=llave_min) 
            #imprimir_tabla(f"Promedio {titulo}", prom_ordenados, es_decimal=True)
            return Error_val
    else:
        print("No hay suficentes datos para evaluar")
        return default


def multiplicar_lista(datos, escalar):
    return [x * escalar for x in datos]


def Sumar_listas(*listas: List[float]) -> List[float]:
    
    if not listas:
        return []

    # Verificar que todas las listas tengan la misma longitud
    longitud = len(listas[0])
    for idx, lst in enumerate(listas, start=1):
        if len(lst) != longitud:
            raise ValueError(
                f"Lista #{idx} tiene longitud {len(lst)}, esperaba {longitud}"
            )

    # Sumar elemento a elemento
    return [sum(vals) for vals in zip(*listas)]
    

def values_offsets(start, stop, Sumas, offsets):
    indices = list(range(start, stop, -1))            # e.g. [9,8,7,6,5,4]
    n = len(indices)                                  # número de términos a promediar
    resultados = []
    for v in offsets:
        s = 0.0
        for i in indices:
            s += (Sumas[-i] + v) / i
        resultados.append(s / n)
    return resultados


def interp_x_from_lists(xx, values, offsets):
    n=len(values)
    #print(xx, " ", values)
    if xx <= values[0]:
        offset_adj = offsets[0]
        return offset_adj, offset_adj, 0, 0.0
    if xx >= values[-1]:
        offset_adj = offsets[-1]
        return offset_adj, offset_adj, n-1, 1.0

    for i in range(n-1):
        v_low = values[i]
        v_high = values[i+1]
        if v_low <= xx <= v_high:
            if v_high == v_low:
                t = 0.0
            else:
                t = (xx - v_low) / (v_high - v_low)
            offset_adj = offsets[i] + t * (offsets[i+1] - offsets[i])
            return offset_adj, offset_adj, i, t

    raise RuntimeError("Intervalo no encontrado; verificar orden y valores")


def mascaidosn(numer, i, j, k, l, m, n):
    nume = numer.tolist() if isinstance(numer, pd.Series) else numer
    aa=probabilidades_bayes(nume, m, n)
    nuu=nume[-i:-j]
    a=probabilidades_bayes(nuu, m, n)
    nuu=nume[-j:]
    b=probabilidades_bayes(nuu, m, n)
    nuu=nume[-k:-l]
    c=probabilidades_bayes(nuu, m, n)
    nuu=nume[-l:]
    d=probabilidades_bayes(nuu, m, n)
    
    return aa, a, b, c, d

def mascaidospx(numer, i, j, k, l, o, p, r, m, n):
    nume = numer.tolist() if isinstance(numer, pd.Series) else numer
    aa=probabilidades_bayes(nume, m, n)
    nuu=nume[-i:-j]
    a=probabilidades_bayes(nuu, m, n)
    nuu=nume[-j:-k]
    b=probabilidades_bayes(nuu, m, n)
    nuu=nume[-k:-l]
    c=probabilidades_bayes(nuu, m, n)
    nuu=nume[-l:-o]
    d=probabilidades_bayes(nuu, m, n)
    nuu=nume[-o:-p]
    e=probabilidades_bayes(nuu, m, n)
    nuu=nume[-p:-r]
    f=probabilidades_bayes(nuu, m, n)
    nuu=nume[-r:]
    g=probabilidades_bayes(nuu, m, n)
    return aa, a, b, c, d, e, f, g

def mascaidosp(numer, i, j, k, l, o, m, n):
    nume = numer.tolist() if isinstance(numer, pd.Series) else numer
    aa=probabilidades_bayes(nume, m, n)
    nuu=nume[-i:-j]
    a=probabilidades_bayes(nuu, m, n)
    nuu=nume[-j:-k]
    b=probabilidades_bayes(nuu, m, n)
    nuu=nume[-k:-l]
    c=probabilidades_bayes(nuu, m, n)
    nuu=nume[-l:-o]
    d=probabilidades_bayes(nuu, m, n)
    nuu=nume[-o:]
    e=probabilidades_bayes(nuu, m, n)
    return aa, a, b, c, d, e


def Lotery(Re, Res):
    banner = [
        # L        OOO      TTTTT    EEEEE   RRRR    Y   Y
        "\n",
        " \t\t\t**          *******      ********     *******     *******      **      ** ",
        " \t\t\t**         **     **        **        **          **   **       **    **  ",
        " \t\t\t**         **     **        **        **          **   **         ****   ",
        " \t\t\t**         **     **        **        ******      ******           **   ",
        " \t\t\t**         **     **        **        **          **   **          **   ",
        " \t\t\t**         **     **        **        **          **    **         **   ",
        " \t\t\t*******     *******         **        *******     **     **        **   ",
        "\n",
    ]
    for line in banner:
        print(f"{Re}{line}{Res}")

def fmt(x, decimals=4):
    s = f"{x:.{decimals}f}"
    if s.startswith("0."):
        return s[1:]           # 0.1234 -> .1234
    if s.startswith("-0."):
        return "-" + s[2:]     # -0.1234 -> -.1234
    return s                  # otros casos (p.ej. 1.0000)


def primer_impres(lista, tip, x, y, z, *cae):
    RED = "\033[31m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    last10 = pd.to_numeric(lista.tail(12), errors="coerce").dropna().astype(float)
    last6 = last10.iloc[-6:]
    def fmt(v, tip):
        v = float(v)
        if tip==0:
            return "{:.0f}".format(v) if v.is_integer() else "{:.1f}".format(v)
        else:
            return "{:.0f}".format(v) if v.is_integer() else "{:.3f}".format(v)
        
    vals = list(map(str, last10))

    colored_parts = []
    for i, v in enumerate(last10.tolist()):   # i es 0-based dentro de last10
        text = fmt(v,tip)
        if i == 4 or i == 8:
            colored_parts.append(f"{RED}{text}{RESET}")
        elif i <= 6:
            colored_parts.append(f"{BLUE}{text}{RESET}")
        elif 5 <= i <= 11:
            colored_parts.append(f"{GREEN}{text}{RESET}")
        else:
            colored_parts.append(text)
    colored = " ".join(colored_parts)   # <- aquí obtienes una sola cadena, no lista
    
    def counts(s):
        mayor = int((s > y).sum())
        return mayor
    
    mayor10 = counts(last10)
    mayor6 = counts(last6)
    P20=lista.mean()
    P30=lista.tolist()
    Prome=sum(P30[-30:])/(len(P30[-30:]))
    imprime_estadisticas(lista, colored, mayor10, mayor6, P20, Prome, x, y, z, tip)
    
    if cae:
        print(aviso_ansi(f"Este numero cayó :{cae}", (18, 35, 140), (255, 170, 170) ))
    #else:
    #     print("Esta es la verdadera ")
    print()


def zonitas(Nume, k):
    print(aviso_ansi("Empezando con Números :", (118, 5, 30), (240, 220, 90) ))
    zonifica = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4
    for v in range(k, 0, -1):
        Numeros=Nume[:-v]
        caida = Nume.iloc[-v]
        cae=zonifica(caida)
        yy=Zonas_Series(Numeros, 0, cae)
        print("")

    #primer_impres(Nume, 0, 2, 4, 7)
    a=Zonas_Series(Nume, 0)
    print(" ".join(f"{v:.3f} " for v in a))
    print("\x1b[48;5;71m  = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   \x1b[0m")
    print("")
    return a


def numemedios(Nume, k):
    Nume2=Lista2_series(Nume)
    for v in range(k, 0, -1):
        Numeros=Nume2[:-v]
        caida = Nume2.iloc[-v]
        primer_impres(Numeros, 0, 1, 3, 5, caida)
        print("                 Numeros Medios ")
        ab=fun_promedios(Numeros, 0, 0, 2)
        Pr_Num, _, _, _, _, rz = procesar_e_imprimir_regresion("Numeros Medios", 0, 4, Numeros, 2, 0,  1, 6)

    primer_impres(Nume2, 0, 1, 3, 5)
    ab=fun_promedios(Nume2, 0, 0, 2) 
    b, _, _, _, _, rz = procesar_e_imprimir_regresion("Numeros Medios", 0, 4, Nume2, 2, 0,  1, 6)
    print(" ".join(f"{v:.3f} " for v in b))
    print("\x1b[48;5;71m= = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   \x1b[0m")
    print("")
    return b


def numerotes(nume, k):
    for v in range(k, 0, -1):
        Numeros=nume[:-v]
        caida = nume.iloc[-v]
        primer_impres(Numeros, 0, 2, 4, 7, caida)
        print("\n                Numeros Completos 0-9 ")
        ab=fun_promedios(Numeros, 0, 0, 1)
        Pr_Num, _, _, _, _, rz = procesar_e_imprimir_regresion("Numeros Completos", 0, 4, Numeros, 2, 0)

    primer_impres(nume, 0, 2, 4, 7)
    ab=fun_promedios(nume, 0, 0, 1) 
    c, _, _, _, _, rz = procesar_e_imprimir_regresion("Numeros Completos", 0, 4, nume, 2, 0)
    print(" ".join(f"{v:.3f} " for v in c))
    print("\x1b[48;5;71m= = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   \x1b[0m")
    print("")
    return c



def total_Numeros(numero, j):

    a=zonitas(numero, j)
    #usuario = input("Por favor, introduce tu nombre: ")
    b=numemedios(numero,j)
    #usuario = input("Por favor, introduce tu nombre: ")
    c=numerotes(numero, j)
    #usuario = input("Por favor, introduce tu nombre: ")
    Resu=[]
    zonificar = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4
    for i, valor in enumerate(c):
        z = zonificar(i)
        Resu.append((3*valor + 2*b[i // 2] + 2*a[z - 1]) / 7)
    #print(Resu)
    return Resu


def zonaciclos(numero, k):
    print("Zonas por frecuencia....")
    for v in range(k, 0, -1):
        Nume=numero[:-v]
        Numeros=pd.Series(Nume)
        caida = numero[-v]
        F_datos=Semanas(Numeros)
        fg=zones_by_freq(Numeros, 0, caida)

    Numeros=pd.Series(numero)
    F_datos=Semanas(Numeros)
    fg=zones_by_freq(Numeros, 0)
    
    return fg


def ciclos(numero, j):
    print(aviso_ansi("Frecuencias en 50 jugadas  :", (118, 5, 30), (240, 220, 90) ))
    jerarquia, Posi = calcular_jerarquias(numero)
    a=zonaciclos(Posi, j)

    for v in range(j, 0, -1):
        
        Numeros=numero[:-v]
        Ultima_Jerarquia=ultima_jerarquia(Numeros)
        jerarquias, Posic = calcular_jerarquias(Numeros)
        F_datos=Semanas(Numeros)
        claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_datos[k]))
        caida = Posi[-v]
        N10 = Posic[-75:]
        nn=pd.Series(N10)
        primer_impres(nn, 0, 2, 5, 8, caida)
        ab=fun_promedios(Posic, 0, 0, 1)
        ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
        
        Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xxx, rz = procesar_e_imprimir_regresion("Jerarquía", 0, 4, Posic, 2, 2, 1, 11)
        print("\x1b[38;5;71m======================================================================================\x1b[0m")
        print()
        
    Ultima_Jerarquia=ultima_jerarquia(numero)
    jerarquias, Posic = calcular_jerarquias(numero)
    F_datos=Semanas(numero)
    claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_datos[k]))
    ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
    N10 = Posic[-75:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 2, 5, 8)
    ab=fun_promedios(Posic, 0, 0, 1)
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xxx, rz = procesar_e_imprimir_regresion("Jerarquía Final", 0, 4, Posic, 2, 2, 1, 11)
    print("\x1b[38;5;71m======================================================================================\x1b[0m")
    print()
        
    nuevos_valores_dict = {}
    errores_dict = {}
    
    for rank, clave in ranking_dict.items():
        nuevo_valor = (Sum14 + rank+1) / xxx
        error = (nuevo_valor - PromGral) / PromGral
        if error < 0:
            error *= -0.999
        
        if rank < 3 :
            error=(3*error+2*a[0])/5
        elif rank < 6 :
            error=(3*error+2*a[1])/5
        else :
            error=(3*error+2*a[2])/5

        nuevos_valores_dict[clave] = nuevo_valor
        errores_dict[clave] = error
        #print(error)

    sorted_keys = sorted(ranking_dict.values())  # esto da [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Pr_Pos_err_ordered = [errores_dict[k] for k in sorted_keys]
    #print(sorted_keys)

    print()
    print(aviso_ansi("\nOrden de jerarquías :", (118, 5, 30), (240, 220, 90) ))
    print("Id\t\t" + "\t".join(str(k) for k in claves_ordenadas))
    print("Repet\t\t" + "\t".join(str(Ultima_Jerarquia[k]) for k in claves_ordenadas))
    print("Apar\t\t" + "\t".join(str(F_datos[k]) for k in claves_ordenadas))
    print("\x1b[38;5;71m===========================================================================================\x1b[0m")
    print("")

    return errores_dict,sorted_keys


def numero_final(nu, err, sorted):
    RED   = "\033[31m"
    RESET = "\033[0m"
    # Calcula los mínimos
    min_jer = min(err.values())
    min_num = min(nu)

    print("\tJerarquías\t\tNumeros\t\tError Num")
    print("*******************************************************************")
    
    ErrorNUm={}
    for k in sorted:
        jer = err[k]
        num = nu[k]
        ErrorNUm[k] = (abs(3*num) + abs(2*jer)) / 5 
        
        # Formatea cada celda, poniendo rojo si coincide con el mínimo
        s_jer = f"{jer:.3f}"
        if jer == min_jer:
            s_jer = f"{RED}{s_jer}{RESET}"
        
        s_num = f"{num:.3f}"
        if num == min_num:
            s_num = f"{RED}{s_num}{RESET}"
        #print(f"{k}\t{s_jer}\t\t")
        print(f"{k}\t{s_jer}\t\t{s_num}\t\t{ErrorNUm[k]:.3f}")

    ErrorOrdenado=ordenar_por_valor(ErrorNUm, ascendente=True)
    #print()
    imprimir_tabla("\nErrores Promedios Numeros ", ErrorOrdenado, es_decimal=True)
    return ErrorOrdenado, ErrorNUm

def Histog3(nume, k, Max):
    
    for v in range(k, 0, -1):
        Histog=obtener_historial_caidas(nume, Max)
        caida = Histog[-v]
        Numeros=Histog[:-v]
        F_datos=Semanas(nume[:-v])
        #print(F_datos)
        H3 = Histo2_con_map(Numeros, 3)
        F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_datos.items()}
        N10=H3[-72:]
        nn=pd.Series(N10)
        
        
        primer_impres(nn, 0, 2, 3, 4)
        print("")
        ab=fun_promedios(H3, 0, 0, 2) 
        #a1=procesar_Histogramas("Histograma 1/3", 0, 0, 1, 25, H3, F_datos_3, 6 )
        #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        a2=procesar_Histogramas("Histograma 1/3", 0, 3, 1, 22, H3, F_datos_3, 6)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        #a3=procesar_Histogramas("Histograma 1/3", 5, 1, 15, H3, F_datos_3, 6)
        #a=escalar_dic(Sumar_diccionarios(a1, a2), 0.3)
        #print(" ".join(f"{v:.3f} " for k, v in a.items()))
        print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    
    Numeros=obtener_historial_caidas(nume, Max)
    F_datos=Semanas(nume)
    #print(F_datos)
    H3 = Histo2_con_map(Numeros, 3)
    F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_datos.items()}
    N10=H3[-72:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 2, 3, 4)
    print("")
    ab=fun_promedios(H3, 0, 0, 2) 
    #a1=procesar_Histogramas("Histograma 1/3", 1, 0, 1, 25, H3, F_datos_3, 6 )
    #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    a=procesar_Histogramas("Histograma 1/3", 0, 3, 1, 22, H3, F_datos_3, 6)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    #a3=procesar_Histogramas("Histograma 1/3", 5, 1, 15, H3, F_datos_3, 6)
    #a=escalar_dic(Sumar_diccionarios(a1, a2), 0.4)
    print(" ".join(f"{v:.3f} " for k, v in a.items()))
    print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    print()
    return a


def Histog2(Nume, k, Max):
    for v in range(k, 0, -1):
        Histog=obtener_historial_caidas(Nume, Max)
        caida = Histog[-v]
        Numeros=Histog[:-v]
        N10=Numeros[-72:]
        nn=pd.Series(N10)
        primer_impres(nn, 0, 3, 7, 11, caida)
        print("")
        F_datos=Semanas(Nume[:-v])
        #print(F_datos)
        H2=Histo2_con_map(Numeros, 2)
        F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_datos.items()}
        N10=H2[-72:]
        nn=pd.Series(N10)
        primer_impres(nn, 0, 2, 4, 6)
        print("") 
        ab=fun_promedios(H2, 0, 0, 2)
        b1=procesar_Histogramas("Histograma 1/2", 0, 0, 1, 25, H2, F_datos_2, 5)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        b2=procesar_Histogramas("Histograma 1/2", 0, 3, 1, 22, H2, F_datos_2, 5)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        #b3=procesar_Histogramas("Histograma 1/2", 5, 1, 15, H2, F_datos_2, 5)
        b=escalar_dic(Sumar_diccionarios(b1, b2), 0.3)
        print(" ".join(f"{v:.3f} " for k, v in b.items()))
        print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
        print("")

    Numeros=obtener_historial_caidas(Nume, Max)
    F_datos=Semanas(Nume)
    #print(F_datos)
    H2 = Histo2_con_map(Numeros, 2)
    F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_datos.items()}
    N10=H2[-72:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 2, 3, 4)
    print("")
    ab=fun_promedios(H2, 0, 0, 2) 
    #a1=procesar_Histogramas("Histograma 1/2", 1, 0, 1, 25, H2, F_datos_2, 5 )
    #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    a=procesar_Histogramas("Histograma 1/2", 0, 3, 1, 22, H2, F_datos_2, 5)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    #a3=procesar_Histogramas("Histograma 1/3", 5, 1, 15, H3, F_datos_3, 5)
    #a=escalar_dic(Sumar_diccionarios(a1, a2), 0.4)
    print(" ".join(f"{v:.3f} " for k, v in a.items()))
    print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    print()
    #at, a1, a2, a3, a4, a5 =mascaidosp(Numeros, 440, 340, 240, 140, 40, 1, 13)
    #imprimir_diccionarios_seleccion([at, a2, a3, a4, a5], F_datos_2, width=6, precision=3)
    
    return a



def Histo_completo(Nume, k, maximo):
    Histog=obtener_historial_caidas(Nume, maximo)
    F_datos=Semanas(Nume)

    for v in range(k, 0, -1):
        caida = Histog[-v]
        Numeros=Histog[:-v]
        N10=Numeros[-70:]
        nn=pd.Series(N10)
        primer_impres(nn, 0, 4, 8, 12, caida)
        #print("")
        F_datos=Semanas(Nume[:-v])
        #print(F_datos)
        #Zonas_Histos(Numeros, 0)
        #print_Histo(Numeros)
        ab=fun_promedios(Numeros, 2, 2, 0)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        e1=procesar_Histogramas("Histograma con 30", 0, 0, 2, 45, Numeros, F_datos, 3)
        #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        #e2=procesar_Histogramas("Histograma con 30", 0, 3, 2, 22, Numeros, F_datos, 3)
        #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        #e3=procesar_Histogramas("Histograma con 30",0, 5, 2, 15, Histog, F_datos, 3)
        #e=escalar_dic(Sumar_diccionarios(e1, e2), 0.4)
        #print(" ".join(f"{v:.3f} " for k, v in e.items()))
        print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
        print("")

    Numeros=obtener_historial_caidas(Nume, maximo)
    N10=Numeros[-70:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 4, 8, 12)
    #print("")
    F_datos=Semanas(Nume)
    #Zonas_Histos(Numeros, 0)
    #print("")
    #ab=fun_promedios(Numeros, 0, 0, 2) 
    a=procesar_Histogramas("Histograma Completo", 0, 3, 1, 22, Numeros, F_datos, 3)
    print(" ".join(f"{v:.3f} " for k, v in a.items()))
    print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    print()
    #print_Histo(Nume)
    """
    at, a1, a2, a3, a4, a5 =mascaidosp(Nume, 225, 175, 125, 75, 25, 1, 11)
    imprimir_diccionarios_tabla([a1, a2, a3, a4, a5], width=8, precision=3)
    print()
    at, a1, a2, a3, a4, a5 =mascaidosp(Histog, 440, 340, 240, 140, 40, 1, 25)
    imprimir_diccionarios_seleccion([at, a2, a3, a4, a5], F_datos, width=6, precision=3)
    """
    return a    


def Llamada_Histo(numero, j, maxi):
    print(aviso_ansi("Histogramas de 1/3 :", (118, 5, 30), (240, 220, 90) ))
    a=Histog3(numero, j, maxi)
    #usuario = input("Por favor, introduce tu nombre: ")    
    print(aviso_ansi("Histogramas de 1/2 :", (118, 5, 30), (240, 220, 90) ))

    b=Histog2(numero, j, maxi)
    #usuario = input("Por favor, introduce tu nombre: ")
    print(aviso_ansi("Histogramas completos :", (118, 5, 30), (240, 220, 90) ))    
    c=Histo_completo(numero, j, maxi)
    
    F_datos=Semanas(numero)
    print_recencia(F_datos)
    
    return a, b, c

def Llamar_Caidas(nume, k):
    print("  --  Posicion de Caidas  --  ")
    print("")
    numero=jerarquia_histo(nume)
    zonifica = lambda x: 1 if x < 4 else 2 if x < 6 else 3 if x < 8 else 4
    lis=[zonifica(x) for x in numero]
    #Imprime_historial(lis, 1, 1, 5)
    print("Posibilidades Zonales por Puesto Semanas")

    for v in range(k, 0, -1):
        Numeros=numero[:-v]
        caida = numero[-v]
        lis=[zonifica(x) for x in Numeros]
        N10 = lis[-70:]
        nn=pd.Series(N10)
        cai=zonifica(caida)
        primer_impres(nn, 0, 1, 2, 3, cai)
        ab=fun_promedios(lis, 0, 0, 2)
        yY, _, _, _, _, rz = procesar_e_imprimir_regresion("Orden Caida", 0, 4, lis, 0, 0, 1, 5)
        #print("\x1b[38;5;71m====     ====     ====     ====     ====     ====     ====     ====\x1b[0m")
        print("")

    lis=[zonifica(x) for x in numero]
    N10 = lis[-75:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 1, 2, 3)
    #ab=fun_promedios(lis, 0, 0, 2)
    a, _, _, _, _, rz = procesar_e_imprimir_regresion("Orden Caida", 0, 4, lis, 0, 0, 1, 5)
    print(" ".join(f"{v:.3f} " for  v in a))
    #print("\x1b[38;5;71m====     ====     ====     ====     ====     ====     ====     ====\x1b[0m")
    print("")

    print("Posibilidades por Puesto Semanas")
    for v in range(k, 0, -1):
        Numeros=numero[:-v]
        caida = numero[-v]
        
        N10 = Numeros[-75:]
        nn=pd.Series(N10)
        primer_impres(nn, 0, 2, 5, 8, caida)
        ab=fun_promedios(Numeros, 0, 0, 1)
        Pb_His2, _, _, _, _, rz = procesar_e_imprimir_regresion("Orden Caida", 0, 4, Numeros, 0, 2, 1, 11)
        print(" ".join(f"{v:.3f} " for  v in Pb_His2))
        print("")

    N10 = numero[-75:]
    nn=pd.Series(N10)
    primer_impres(nn, 0, 2, 5, 8)
    ab=fun_promedios(numero, 0, 0, 1)
    b, _, _, _, _, rz = procesar_e_imprimir_regresion("Orden Caida", 0, 4, numero, 0, 2, 1, 11)
    print(" ".join(f"{v:.3f} " for  v in b))
    print("\x1b[38;5;71m====     ====     ====     ====     ====     ====     ====     ====     ====     ====\x1b[0m")
    print("")
    res = [b[i] + a[zonifica(i + 1) - 1] for i in range(len(b))]
    #print(a)
    #print(b) 
    #print(res)

    at, a1, a2, a3, a4, a5 =mascaidosp(numero, 225, 175, 125, 75, 25, 1, 11)
    imprimir_diccionarios_tabla([a1, a2, a3, a4, a5], width=8, precision=3)
    return res


def imprimir_diccionarios_tabla(dicts, width=6, precision=2, keys=None, rows_per_block=2):

    if not dicts:
        return
    if keys is None:
        keys = sorted(dicts[0].keys(), key=int)

    # formato: cabecera centrada, valores a la derecha con precision fija
    header_fmt = " ".join(f"{{:^{width}}}" for _ in keys)
    #val_fmt = " ".join(f"{{:>{width}.{precision}f}}" for _ in keys)
    val_fmt = " ".join(f"{{:>{width}}}" for _ in keys)


    # imprimir cabecera y separador
    header_vals = [str(k) for k in keys]
    print(header_fmt.format(*header_vals))
    print(" ".join("-" * width for _ in keys))

    for i, d in enumerate(dicts):
        #row_vals = [float(d[k]) for k in keys]
        #print(val_fmt.format(*row_vals))
        row_vals = [f"{float(d[k]):.{precision}f}".lstrip("0") for k in keys]
        print(val_fmt.format(*row_vals))

        if (i + 1) % rows_per_block == 0 and (i + 1) != len(dicts):
            print()
    print()


def imprimir_diccionarios_seleccion(dicts, selector, width=6, precision=3, rows_per_block=2, missing_fill=0.0):
    if not dicts:
        return

    # construir lista de claves objetivo en el orden del selector (mantener como el tipo original del dict)
    # tomamos el primer dict para decidir si sus claves son str o int
    first = dicts[0]
    sample_is_str = all(isinstance(k, str) for k in first.keys())

    # obtener valores del selector en orden de su key (0,1,2,...)
    ordered_selector_values = [v for _, v in sorted(selector.items(), key=lambda x: int(x[0]))]

    # normalizar targets al tipo de las claves en los dicts
    if sample_is_str:
        targets = [str(v) for v in ordered_selector_values]
    else:
        targets = [int(v) for v in ordered_selector_values]

    # formatos
    header_fmt = " ".join(f"{{:^{width}}}" for _ in targets)
    val_fmt = " ".join(f"{{:>{width}}}" for _ in targets)

    # imprimir cabecera con las claves reales (targets)
    header_vals = [str(t) for t in targets]
    print(header_fmt.format(*header_vals))
    print(" ".join("-" * width for _ in targets))

    # imprimir filas; usar missing_fill si la clave falta
    for i, d in enumerate(dicts):
        row_strs = []
        for t in targets:
            # intentar extraer valor con d.get para evitar KeyError
            raw = d.get(t, None) if isinstance(d, dict) else None
            if raw is None:
                # intentar versión opuesta de tipo (por ejemplo buscar '6' si t es 6)
                alt_key = str(t) if not isinstance(t, str) else int(t)
                raw = d.get(alt_key, None)
            if raw is None:
                v = float(missing_fill)
            else:
                v = float(raw)
            s = f"{v:.{precision}f}".lstrip("0")
            row_strs.append(s)
        print(val_fmt.format(*row_strs))

        if (i + 1) % rows_per_block == 0 and (i + 1) != len(dicts):
            print()
    print()





def Imprime_historial(numero, Preg, x, y):
    at, a1, a2, a3, a4 =mascaidosn(numero, 40, 15, 20, 10, x, y)
    imprimir_diccionarios_tabla([a1, a2, a4], width=7, precision=2)
    
    if Preg==1:
        at, a1, a2, a3, a4, a5, a6, a7 =mascaidospx(numero, 325, 275, 225, 175, 125, 75, 25, x, y)
        imprimir_diccionarios_tabla([a1, a2, a3, a4, a5, a6, a7], width=7, precision=2)

def valores_estadisticas(l, n, x1, x2, x3):
    prom=l.tail(n).mean()
    s9=l.tail(n-1).sum()
    S1=(s9 + x1)/n
    S2=(s9 + x2)/n
    S3=(s9 + x3)/n
    valores1 = [0.0, S1, S2, S3]
    mayores1 = [(i,v) for i, v in enumerate(valores1) if v >= prom]
    m_1 = min(mayores1, key=lambda x: x[1]) if mayores1 else (4, None)
    #m2=sum(mayores1)/len(mayores1)
    m1 = m_1[0] 
    print("{:.2f}".format(prom), "{", "{:.2f}".format(S1), " * ", "{:.2f}".format(S2)," * ", "{:.2f}".format(S3), "}{", f"{int(m1)}", "}", end="\t")
    return m1


def valores_estadisticas5(l, n, er):
    
    prom=l.tail(n).mean()
    s9=l.tail(n-1).sum()
    Ss=[(s9 + x)/n for x in er]
    
    mayores1 = [(i,v) for i, v in enumerate(Ss) if v > prom]
    m_1 = min(mayores1, key=lambda x: x[1]) if mayores1 else (7, None)
    m1 = m_1[0] + 1
    
    #print(fmt(prom), "{", " * ".join(fmt(s) for s in Ss), "}{", f"{int(m1)}", "}")
    print("{:.4f}".format(prom), "{", "{:.0f}".format(m1), "}", end="\t")

    return m1


def imprime_estadisticas(lista, colored_all, mayor10, mayor6, Prom1, Prom2, X1, X2, X3, tipo):
    print(f"Lista Actual {colored_all}      M10= {mayor10}  M6= {mayor6}      Prom {Prom2:.4f}")
    l3=[]
    l5=[]
    if tipo==0:
        #print("\t", end = " ")
        er=[X1, X2, X3]
        a=valores_estadisticas(lista, 10, X1, X2, X3)
        l3.append(a)
        l3.append(a)
        b=valores_estadisticas(lista, 8, X1, X2, X3)
        l3.append(b)
        c=valores_estadisticas(lista, 4, X1, X2, X3)
        l3.append(c)
        y=sum(l3)/len(l3)
    
    elif tipo==1:
        err=[-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        a=valores_estadisticas5(lista, 20, err)
        l5.append(a)
        b=valores_estadisticas5(lista, 5, err)
        l5.append(b)
        #print()
        c=valores_estadisticas5(lista, 8, err)
        l5.append(c)
        l5.append(c)
        d=valores_estadisticas5(lista, 4, err)
        l5.append(d)
        y=sum(l5)/len(l5)
    else:
        err=[-0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        a=valores_estadisticas5(lista, 50, err)
        l5.append(a)
        b=valores_estadisticas5(lista, 45, err)
        l5.append(b)
        #print()
        c=valores_estadisticas5(lista, 40, err)
        l5.append(c)
        #l5.append(c)
        d=valores_estadisticas5(lista, 35, err)
        l5.append(d)
        y=sum(l5)/len(l5)
    
    print("P.Total: ", "{:.2f}".format(y))
    
    if tipo==2:
        PromAxl(lista, Prom2, tipo)
    else:
        PromAxl(lista, Prom1, tipo)


def PromAxl(Lista, p, tipo):
    # colores ANSI
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    l=Lista.tolist()
    a=sum(l[-15:])
    a=p*16-a
    b=sum(l[-12:])
    b=p*13-b
    c=sum(l[-9:])
    c=p*10-c
    d=sum(l[-6:])
    d=p*7-d
    e=sum(l[-3:])
    e=p*4-e
    
    if tipo>0:
        if (a>0.12) or (a<-0.12):
            a=a/6
        if (b>0.12) or (b<-0.12):
            b=b/6
        if (c>0.12) or (c<-0.12):
            c=c/6
        if (d>0.12) or (d<-0.12):
            d=d/6
        if (e>0.12) or (e<-0.12):
            e=e/6

    else:
        if a<0 :
            a=1
        if b<0 :
            b=1
        if c<0 :
            c=1
        if d<0 :
            d=1
        if e<0 :
            e=1
    
    difs, prr = generar_diferencias(Lista, tipo)
    pr10=sum(difs[-10:])/len(difs[-10:])
    pr4=sum(difs[-4:])/len(difs[-4:])
    print("")
    last10 = difs[-10:]

    # construir lista con coloreado en índice 1 (2º) e índice 5 (6º)
    formatted = []
    for i, x in enumerate(last10):
        if tipo>0:
            s = f"{x:.3f}"
        else:
            s = f"{x:.1f}"

        if i == 3:
            s = f"{GREEN}{s}{RESET}"
        elif i == 6:
            s = f"{GREEN}{s}{RESET}"
        formatted.append(s)

    if tipo>0:
        prome=(a+b+c+d+e)/5
        prome4=(b+c+d+e)/4
        prome3=(c+d+e)/3
        promem=(b+c+d)/3
        prome1=error_to_scale(prome)
        prome14=error_to_scale(prome4)
        prome13=error_to_scale(prome3)
        promem1=error_to_scale(promem)
        print(f"{a:.3f}  {b:.3f}  {c:.3f}  {d:.3f}  {e:.3f}      {{{prome1:.1f}  {prome14:.1f}  {prome13:.1f}  {promem1:.1f}}}     {{{prome:.3f}  {prome4:.3f}  {prome3:.3f}  {promem:.3f}}}")
        #print("{ " + ",  ".join(f"{x:.3f}" for x in difs[-10:]) + " }", end="")
        print("{ " + ", ".join(formatted) + " }", end="")
    else :
        prome=(a+b+c)/3
        prome4=(a+b+c+d)/4
        prome3=(b+c+d)/3
        print(f"{a:.0f}  {b:.0f}  {c:.0f}  {d:.0f}  {e:.0f}         {prome:.0f}   {prome4:.0f}   {prome3:.0f}")
        #print("{ " + ",  ".join(f"{x:.1f}" for x in difs[-10:]) + " }", end=" ")
        print("{ " + ", ".join(formatted) + " }", end="")
    print(f"  {{{prr:.3f}  {pr10:.3f}  {pr4:.3f}}}", end="   ")
    calcular_Axl(last10, tipo, prr, pr10, pr4)



def fun_promedios(lista, modo, t, h=3):
    RESET     = "\033[0m"
    FG_RED    = "\033[31m"
    FG_BLUE   = "\033[34m"
    BG_GREEN  = "\033[107m"
    FG_DEFAULT= "\033[39m"

    ultimos = lista[-100:]
    PromT = mean(ultimos)
    if h==3:
        #n10=lista[-20:]
        nn=pd.Series(ultimos)
        n=nn.mean()
        primer_impres(nn, t, -0.04, 0, 0.04)
       
    if h==0:
        inic, medio, fin1, fin2, fin=3, 6, 9, 12, 15
    elif h==1:
        inic, medio, fin1, fin2, fin=1, 3, 5, 7, 9
    elif h==2:
        inic, medio, fin1, fin2, fin=1, 2, 3, 4, 5
    else:
        inic=-0.08
        medio=-0.04
        fin1=0
        fin2=0.04
        fin=0.08

    # Generar bloques coloreados o formateados
    bloques = []
    for val in ultimos[-12:]:
        if modo == 1:
            texto = str(val)
            bloque = f"{BG_GREEN}{FG_DEFAULT}{texto}{RESET}"
        elif modo == 0:
            bloque = f"{val:.0f}"
        else:
            texto = f"{val:.3f}"
            if val < 0:
                bloque = f"{FG_RED}{texto}{RESET}"
            elif val > 0:
                bloque = f"{FG_BLUE}{texto}{RESET}"
            else:
                bloque = texto
        bloques.append(bloque)

    # 2. Creamos grupos de 3 con dos espacios entre números
    group_size = 3
    grupos = ["  ".join(bloques[i:i+group_size])
            for i in range(0, len(bloques), group_size)]

    def strip_ansi(s):
        return re.sub(r"\x1b\[[0-9;]*m", "", s)
    if modo==2:
        tab_width = 4
    else:
        tab_width = 3

    max_len = max(len(strip_ansi(g)) for g in grupos)
    fixed_width = ((max_len + tab_width - 1) // tab_width) * tab_width
    grupos_pad = [g + " " * (fixed_width - len(strip_ansi(g))) for g in grupos]
    salida = "\t".join(grupos_pad)
    print(salida)
    # Funciones auxiliares
    avg = lambda n: mean(lista[-n:])         # promedio de los últimos n elementos
    sum_n = lambda n: sum(lista[-n:])        # suma de los últimos n elementos
    # Cálculos principales
    
    p9, p8, p7, p6, p5, p4, p3, p2 = (avg(n) for n in (9, 8, 7, 6, 5, 4, 3, 2))
    s1, s2, s3, s4, s5, s6, s7, s8  = sum_n(1), sum_n(2), sum_n(3), sum_n(4), sum_n(5), sum_n(6), sum_n(7), sum_n(8)
    promed=[p9, p8, p7, p6, p5, p4, p3, p2, 0.0]
    offsets = [inic, medio, fin1, fin2, fin]
    Sumas=[s8, s7, s6, s5, s4, s3, s2, s1, 0.0]
    listax=[]

    x9=sum(promed[-9:-3])/len(promed[-9:-3])
    x8=sum(promed[-9:-5])/len(promed[-9:-5])
    x7=sum(promed[-7:-3])/len(promed[-7:-3])
    x6=sum(promed[-5:-2])/len(promed[-5:-2])
    x5=sum(promed[-5:-1])/len(promed[-5:-1])

    vo1=values_offsets(9, 3, Sumas,  offsets)
    x_final1, offset_ajustado, index_low, tx = interp_x_from_lists(x9, vo1, offsets)
    listax.append(x_final1)
    
    vo2=values_offsets(9, 5, Sumas, offsets)
    x_final2, offset_ajustado, index_low, tx = interp_x_from_lists(x8, vo2, offsets)
    listax.append(x_final2)

    vo3=values_offsets(7, 3, Sumas, offsets)
    x_final3, offset_ajustado, index_low, tx = interp_x_from_lists(x7, vo3, offsets)
    listax.append(x_final3)

    vo4=values_offsets(5, 2, Sumas, offsets)
    x_final4, offset_ajustado, index_low, tx = interp_x_from_lists(x6, vo4, offsets)
    listax.append(x_final4)

    vo5=values_offsets(5, 1, Sumas, offsets)
    x_final5, offset_ajustado, index_low, tx = interp_x_from_lists(x5, vo5, offsets)
    listax.append(x_final5)

    xxx=sum(listax)/len(listax)
    xyy=(x9 + x8 + x7 + x6 + x5 +p3)/6
    xxy=(sum(listax) - min(listax) - max(listax))/3
    listax.append(x_final4)
    listax.append(x_final5)
    yyy=sum(listax)/len(listax)

    print(
        "\t".join(colorear(v, lbl, 1) for v, lbl in 
                zip(( x9, x8, x7, x6, x5), ("P9-4", "P9-6", "P7-4", "P5-3", "P5-2")))
    )
    #print(
    #    "\t".join(colorear(v, lbl, 1) for v, lbl in 
    #            zip((x_final1, x_final2, x_final3, x_final4, x_final5), ("X1", "X2", "X3", "X4", "X5")))
    #)
    print(
        "\t".join(colorear(v, lbl, 1) for v, lbl in 
                zip((PromT, xxx, xyy, yyy, xxy), ("Prom", "P5", "P P", "PRU", "MIO")))
    )
    print()
    return xxy


def imprime_dict_color(d, decimals=2, gap=4, separator_width=3):
    # preparar datos
    keys_original = list(d.keys())
    vals_original = [d[k] for k in keys_original]

    # izquierda: orden por clave
    keys_sorted = sorted(keys_original)
    vals_sorted = [d[k] for k in keys_sorted]

    # decidir índices a colorear respecto al orden original
    n_top = min(2, len(vals_original))
    n_bottom = min(4, max(0, len(vals_original) - n_top))

    enumerados = list(enumerate(vals_original))
    top_sorted = sorted(enumerados, key=lambda x: (x[1], -x[0]), reverse=True)
    top_indices = {idx for idx, _ in top_sorted[:n_top]}

    bottom_candidates = [e for e in enumerados if e[0] not in top_indices]
    bottom_sorted = sorted(bottom_candidates, key=lambda x: (x[1], x[0]))
    bottom_indices = {idx for idx, _ in bottom_sorted[:n_bottom]}

    # formato de texto de valores
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    left_keys = [str(k) for k in keys_sorted]
    left_vals = [fmt(v) for v in vals_sorted]

    right_keys = [str(k) for k in keys_original]
    right_vals = [fmt(v) for v in vals_original]

    # anchos por columna (usar ancho máximo por columna entre ambas tablas)
    # calculamos ancho por columna como max(len(key), len(val))
    col_widths_left = [max(len(k), len(v)) for k, v in zip(left_keys, left_vals)]
    col_widths_right = [max(len(k), len(v)) for k, v in zip(right_keys, right_vals)]

    # construir filas para cada tabla (clave en una fila, valor en otra)
    # función que alinea y aplica color en la fila de valores de la tabla derecha
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # construir texto de tabla horizontal (clave-row y valor-row) dado keys, vals, col_widths, opcional color_indices
    def build_table_rows(keys, vals, col_widths, color_indices=None, is_right=False):
        # header (keys)
        header_cells = [k.center(w) for k, w in zip(keys, col_widths)]
        header = " ".join(header_cells)
        # values
        val_cells = []
        for i, (v, w) in enumerate(zip(vals, col_widths)):
            text = v.center(w)
            if is_right and color_indices is not None:
                if i in color_indices['top']:
                    text = f"{RED}{text}{RESET}"
                elif i in color_indices['bottom']:
                    text = f"{GREEN}{text}{RESET}"
            val_cells.append(text)
        vals_row = " ".join(val_cells)
        return header, vals_row

    # si el número de columnas difiere, igualamos rellenando con espacios
    max_cols = max(len(left_keys), len(right_keys))
    def pad_list(lst, length, pad=""):
        return lst + [pad] * (length - len(lst))

    left_keys_p = pad_list(left_keys, max_cols, "")
    left_vals_p = pad_list(left_vals, max_cols, "")
    right_keys_p = pad_list(right_keys, max_cols, "")
    right_vals_p = pad_list(right_vals, max_cols, "")

    col_widths_left_p = pad_list(col_widths_left, max_cols, 0)
    col_widths_right_p = pad_list(col_widths_right, max_cols, 0)

    # ajustar anchos mínimos (si 0 -> tomar ancho de header/val)
    for i in range(max_cols):
        w_left = col_widths_left_p[i] or max(len(left_keys_p[i]), len(left_vals_p[i]))
        w_right = col_widths_right_p[i] or max(len(right_keys_p[i]), len(right_vals_p[i]))
        col_widths_left_p[i] = w_left
        col_widths_right_p[i] = w_right

    # construir filas de ambas tablas
    left_header, left_vals_row = build_table_rows(left_keys_p, left_vals_p, col_widths_left_p)
    right_header, right_vals_row = build_table_rows(
        right_keys_p,
        right_vals_p,
        col_widths_right_p,
        color_indices={'top': top_indices, 'bottom': bottom_indices},
        is_right=True
    )

    # separación vertical entre tablas y separador central
    sep_between_tables = " " * gap
    vertical_bar = " " * separator_width + "│" + " " * separator_width

    # calcular líneas de separación horizontales (guiones) con la longitud de cada tabla
    table_left_width = sum(col_widths_left_p) + (max_cols - 1)  # espacios entre columnas
    table_right_width = sum(col_widths_right_p) + (max_cols - 1)

    bar_left = "-" * table_left_width
    bar_right = "-" * table_right_width

    # imprimir: línea de guiones, headers lado a lado, valores lado a lado, línea de guiones
    print(f"{bar_left}{sep_between_tables}{vertical_bar}{sep_between_tables}{bar_right}")
    print(f"{left_header}{sep_between_tables}{vertical_bar}{sep_between_tables}{right_header}")
    print(f"{left_vals_row}{sep_between_tables}{vertical_bar}{sep_between_tables}{right_vals_row}")
    print(f"{bar_left}{sep_between_tables}{vertical_bar}{sep_between_tables}{bar_right}")


def imprime_dict_color_horizontal_both(d, decimals=3, gap=2, separator_width=1):
    # preparar datos
    keys_original = list(d.keys())
    vals_original = [d[k] for k in keys_original]
    # izquierda: orden por clave
    keys_sorted = sorted(keys_original)
    vals_sorted = [d[k] for k in keys_sorted]
    # elegir índices de los 2 más pequeños y los siguientes 2 (sobre el orden original)
    n = len(vals_original)
    if n == 0:
        print()
        return

    enumerados = list(enumerate(vals_original))
    # ordenar ascendente por valor, desempatar por índice
    asc = sorted(enumerados, key=lambda x: (x[1], x[0]))
    green_indices = {idx for idx, _ in asc[:2]}               # los 2 más pequeños
    yellow_indices = {idx for idx, _ in asc[2:4]}             # los 2 siguientes
    red_indices = {idx for idx, _ in asc[-2:]}

    # formato de texto de valores
    def fmt(v):
        if isinstance(v, float):
            s = f"{v:.{decimals}f}"
            # Si es negativo y comienza con -0. -> "-.xxx"
            if s.startswith("-0."):
                return "-" + s[2:]
            # Si es positivo y comienza con 0. -> ".xxx" (incluye 0.000 -> .000)
            if s.startswith("0."):
                return s[1:]
            return s
        return str(v)

    left_keys = [str(k) for k in keys_sorted]
    left_vals = [fmt(v) for v in vals_sorted]
    right_keys = [str(k) for k in keys_original]
    right_vals = [fmt(v) for v in vals_original]

    # mapear para la tabla izquierda: índice en orden original de cada clave ordenada
    map_left_idx = []
    for k in keys_sorted:
        # intentar convertir clave a int si aplica, sino buscar por string
        try:
            key_cast = int(k)
        except Exception:
            key_cast = k
        map_left_idx.append(keys_original.index(key_cast) if key_cast in keys_original else keys_original.index(k))

    # anchos por columna
    col_widths_left = [max(len(k), len(v)) for k, v in zip(left_keys, left_vals)]
    col_widths_right = [max(len(k), len(v)) for k, v in zip(right_keys, right_vals)]
    # colores ANSI
    GREEN   = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    RESET = "\033[0m"
    # construir filas de una tabla horizontal (header y fila de valores), aplicando mapa a índices originales si se da
    def build_table_rows(keys, vals, col_widths, index_map=None):
        header_cells = [k.center(w) for k, w in zip(keys, col_widths)]
        header = "  ".join(header_cells)
        val_cells = []
        for i, (v, w) in enumerate(zip(vals, col_widths)):
            text = v.center(w)
            orig_idx = index_map[i] if index_map is not None else i
            if orig_idx in green_indices:
                text = f"{GREEN}{text}{RESET}"
            elif orig_idx in yellow_indices:
                text = f"{YELLOW}{text}{RESET}"
            elif orig_idx in red_indices:
                text = f"{RED}{text}{RESET}"
                
            val_cells.append(text)
        vals_row = "  ".join(val_cells)
        return header, vals_row
    # normalizar columnas para que ambas tablas tengan el mismo número de "columnas" al imprimir
    max_cols = max(len(left_keys), len(right_keys))
    def pad_list(lst, length, pad=""):
        return lst + [pad] * (length - len(lst))

    left_keys_p = pad_list(left_keys, max_cols, "")
    left_vals_p = pad_list(left_vals, max_cols, "")
    right_keys_p = pad_list(right_keys, max_cols, "")
    right_vals_p = pad_list(right_vals, max_cols, "")
    col_widths_left_p = pad_list(col_widths_left, max_cols, 0)
    col_widths_right_p = pad_list(col_widths_right, max_cols, 0)
    map_left_idx_p = pad_list(map_left_idx, max_cols, None)

    # ajustar anchos mínimos
    for i in range(max_cols):
        col_widths_left_p[i] = col_widths_left_p[i] or max(len(left_keys_p[i]), len(left_vals_p[i]))
        col_widths_right_p[i] = col_widths_right_p[i] or max(len(right_keys_p[i]), len(right_vals_p[i]))

    # construir filas
    left_header, left_vals_row = build_table_rows(left_keys_p, left_vals_p, col_widths_left_p, index_map=map_left_idx_p)
    right_header, right_vals_row = build_table_rows(right_keys_p, right_vals_p, col_widths_right_p, index_map=None)

    # separación y barras
    sep_between_tables = " " * gap
    vertical_bar = " " * separator_width + "│" + " " * separator_width

    table_left_width = sum(col_widths_left_p) + (max_cols + 8)
    table_right_width = sum(col_widths_right_p) + (max_cols + 8)
    bar_left = "-" * table_left_width
    bar_right = "-" * table_right_width

    # imprimir
    print(f"{bar_left}{sep_between_tables}{vertical_bar}{sep_between_tables}{bar_right}")
    print(f"{left_header}{sep_between_tables}{vertical_bar}{sep_between_tables}{right_header}")
    print(f"{left_vals_row}{sep_between_tables}{vertical_bar}{sep_between_tables}{right_vals_row}")
    print(f"{bar_left}{sep_between_tables}{vertical_bar}{sep_between_tables}{bar_right}")


def generar_diferencias(lista, t):
    if t>0:
        ta=4
        tp=25
        tpf=25.0
    else:
        ta=3
        tp=25
        tpf=25.0

    n = len(lista)
    resultados = []
    start_i = n - 20
    end_i = n - 2  # inclusive en el recorrido pero usamos elemento siguiente para la resta
    # validaciones mínimas
    if n < 56:
        raise ValueError("La lista debe tener al menos 36 elementos para calcular ventanas de 30 y 6.")
    
    for i in range(start_i, end_i + 1):  # recorre desde -6 hasta -2 (índices positivos)
        # suma de los 6 valores anteriores a i: índices [i-6, i) -> corresponde a slice lista[i-6:i]
        suma_6 = sum(lista[i-ta:i])
        # promedio de los 30 valores anteriores a i: índices [i-30, i)
        ventana_30 = lista[i-tp:i]
        if len(ventana_30) < tp:
            raise ValueError(f"En el índice {i} no hay 30 valores anteriores disponibles.")
        promedio_30 = sum(ventana_30) / tpf

        # cálculo del pronóstico: promedio_30 * 7 - suma_6, con clipping entre 1 y 10
        pronostico = promedio_30 * (ta+1) - suma_6
        
        if t>0:
            if pronostico < -0.1:
                pronostico = -0.03
            elif pronostico > 0.1:
                pronostico = 0.03

        else:
            if pronostico < 0:
                pronostico = 1
            elif pronostico > 10:
                pronostico = 10

        # siguiente en la lista (comparar con el siguiente elemento)
        siguiente = lista[i+1]
        # restar al pronóstico el siguiente valor y guardar en resultados
        diferencia = pronostico - siguiente
        resultados.append(diferencia)
        pr=sum(resultados)/len(resultados)

    return resultados, pr


def calcular_Axl(lista, t, a, b, c):
    # Validación básica de tipos
    try:
        a = float(a); b = float(b); c = float(c)
    except Exception:
        raise ValueError("a, b y c deben ser números")

    if not isinstance(lista, (list, tuple)):
        raise ValueError("lista debe ser una lista o tupla de números")
    # Convertir elementos de lista a float cuando sea posible
    nums = []
    for i, v in enumerate(lista):
        try:
            nums.append(float(v))
        except Exception:
            raise ValueError(f"Elemento en lista con índice {i} no es convertible a número")

    # Sumas de los últimos elementos (si hay menos elementos usa lo que haya)
    slast6 = sum(nums[-6:]) if len(nums) > 0 else 0.0
    slast3 = sum(nums[-3:]) if len(nums) > 0 else 0.0
    # Cálculos intermedios
    #avg1 = (a + b) / 2.0
    #avg2 = (a + c) / 2.0
    valA = a * 7.0 - slast6
    valB = a * 4.0 - slast3
    valor1 = valA
    valor2 = (valA + valB)/2
    valor3 = valB
    RED = "\033[91m"
    RESET = "\033[0m"

    if t>0:
        fmt = lambda v: f"{v:.3f}"
    else:
        fmt = lambda v: f"{v:.2f}"

    s1 = fmt(valor1)
    s2 = f"{RED}{fmt(valor2)}{RESET}"
    s3 = fmt(valor3)

    """
    v1=max(valA, valB)
    v2=min(valA, valB)
    # Diferencias y error relativo
    x = a - b
    y = a - c
    # Manejo si y es cero (evitar división por cero)
    if y == 0:
        # Si y == 0, definimos error_rel como infinito si x != 0, o 0 si ambos son 0
        if x == 0:
            error_rel = 0.0
        else:
            error_rel = float('inf')  # representará que la diferencia es enorme
    else:
        error_rel = (x - y) / y
    # Determinar cuál resaltar
    condicion_grande = abs(error_rel) > 0.15
    
    # Lógica de resaltado según reglas
    if condicion_grande:
        if y > x:
            # resaltar primer valor (valor1)
            s1 = f"{RED}{fmt(valor1)}{RESET}"
            s2 = fmt(valor2)
            s3 = fmt(valor3)
            motivo = "x > y (error relativo > 0.2), se resalta valor1"
        else:
            # y > x, resaltar tercer valor (valor3)
            s1 = fmt(valor1)
            s2 = fmt(valor2)
            s3 = f"{RED}{fmt(valor3)}{RESET}"
            motivo = "y > x (error relativo > 0.2), se resalta valor3"
    else:
        # no hay "mas grande que el otro", resaltar valor2
        s1 = fmt(valor1)
        s2 = f"{RED}{fmt(valor2)}{RESET}"
        s3 = fmt(valor3)
        motivo = "no hay diferencia relativa > 0.2, se resalta valor2"
    """

    print(f"{{{s1} {s2} {s3}}}")

    


def main(file_path):
    # Códigos ANSI para rojo y reset
    RED   = "\033[31m"
    RESET = "\033[0m"
    #Lotery(RED, RESET)
    print(aviso_ansi("\nE M P E Z A M O S.....", (20, 210, 100), (220, 120, 20) ))
    Inic = datetime.datetime.now().strftime("%H:%M:%S")
    veces=0
    Maximo=24
    Numero = leer_datos_excel(file_path)
    numb3rs=total_Numeros(Numero, veces)
    errores_dict, sorted_keys=ciclos(Numero, veces)
    numbers, err =numero_final(numb3rs, errores_dict, sorted_keys) 
    Imprime_historial(Numero, 1, 0, 10)
    print(aviso_ansi("\nTERMINAMOS NUMEROS...", (220, 110, 10), (120, 220, 200) ))
    print("")
    print("                   ***   Recencia de Semanas   ***")
    Cai=Llamar_Caidas(Numero, veces)
    F_datos=Semanas(Numero)
    h3, h2, h =Llamada_Histo(Numero, veces, Maximo)
    ref = h
    keys = list(ref.keys())
    cai = dict(zip(keys, Cai))
    dicts = [h3, h2, h, h] 
    H_histo = {k: sum(abs(d[k]) for d in dicts) for k in keys}
    numb3rs_d = dict(zip(range(10), numb3rs))
    Ordenados=ordenar_por_valor(H_histo, ascendente=True)
    imprimir_tabla("Probabilidad Caidas Histograma ", H_histo, es_decimal=True)
    print("")
    imprime_dict_color_horizontal_both(numbers)
    imprime_dict_color_horizontal_both(numb3rs_d)
    imprime_dict_color_horizontal_both(errores_dict)
    imprime_dict_color_horizontal_both(cai)
    imprime_dict_color_horizontal_both(H_histo)
    #numbers_d = dict(zip(range(10), numbers))
    #imprime_dict_color_horizontal_both(numbers_d)
    imprimir_tabla("Probabilidad Caidas Histograma ", Ordenados, es_decimal=True)
    print("")
    print(aviso_ansi("\nTERMINAMOS HISTOGRAMAS \n", (220, 110, 10), (120, 220, 200) ))
    print("")
    print("  --  Pseudo probabilidad jerárquica  --  ")
    print("")
    Prior1=calcular_alpha_prior(Numero)
    Prior=ordenar_por_valor(Prior1, ascendente=False)
    
    Probab_mayor = aplicar_regresion_logistica_mayor_menor(Numero)
    if Probab_mayor is not None:
        print(f"\nProbabilidad (Reg. Logística) de que el siguiente número sea mayor que 4: {Probab_mayor:.4f}")

    Probab_par = aplicar_regresion_logistica_par_impar(Numero)
    if Probab_par is not None:
        print(f"Probabilidad (Reg. Logística) de que el siguiente número sea par: {Probab_par:.4f}")
    print()

    """
    imprimir_tabla("Probabilidad de Numeros ", numbers, es_decimal=True)
    print("")
    imprimir_tabla("Posicion de Caida ", Cai, es_decimal=True)
    imprimir_tabla("Probabilidad de Histograma ", H_histo, es_decimal=True)
    print(aviso_ansi(f"\nTERMINAMOS AQUI : ", (118, 5, 30), (240, 220, 100) ))
    print(Inic)
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    
    
    print("  --  Probabilidades recencia números  siguientes --  ")
    print("")
    #HIs_Sig=LLama_Sigui(Numeros)
    """
    
    #usuario = input("Por favor, introduce tu nombre: ")
    

if __name__ == "__main__":
    print("Hello World")
    file_path = 'D:/loter.xlsx'
    main(file_path)