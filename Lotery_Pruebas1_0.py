import pandas as pd
import numpy as np
import datetime
import math
from statistics import mean
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict, deque
from EstadisticasLotery import aplicar_svr, last_interval_expand, probabilidades_bayes, filtrar_segmentos_df, print_recencias
from Graficas_tripletas import plot_jugadas


def print_Histo(seq: List[int], k: int = 3) -> None:
    print("Probemos...")
    print("Probemos...")
    try:
        last_pos = [-1] * 10
        gaps = [deque(maxlen=k) for _ in range(10)]
        
        for i, v in enumerate(seq):
            print("i,v:", i, v)           # debug
            if not isinstance(v, int):
                print("Valor no entero:", v); return
            if v < 0 or v >= 10:
                print("Valor fuera de rango 0-9:", v); return
            lp = last_pos[v]
            if lp >= 0:
                gaps[v].append(i - lp)
            last_pos[v] = i

        print("")
        for d in range(10):
            print(f"{d}: {list(gaps[d])}")
        print("")
    except Exception as e:
        print("Excepción en print_Histo:", type(e).__name__, e)


def Zonas_Series(lista, Preg, *c):
    zonificar = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4
    zonas = lista.map(zonificar).astype(int)

    if c:
        a1, a5, Pp, P = primer_impres(zonas, 0, 1, 2, 3, c[0])    
    else:
        a1, a5, Pp, P = primer_impres(zonas, 0, 1, 2, 3)
        
    print("")
    Pr2, _, _, _, _, rz  = procesar_e_imprimir_regresion(Preg, 4, zonas, 2, 4, 1, 5)

    return Pr2


def zones_by_freq(serie: List[int], Preg, *cae) -> List[int]:
    zonifica = lambda x: 1 if x < 4 else 2 if x < 6 else 3 if x < 8 else 4
    zonas = [zonifica(x) for x in serie]
    
    N10 = zonas[-90:]
    zona=pd.Series(N10)
    if cae:
        c=zonifica(cae[0])
        a1, a5, Pp, P = primer_impres(zona, 0, 1, 2, 3, c)
    else:
        a1, a5, Pp, P = primer_impres(zona, 0, 1, 2, 3)
    
    print("")
    Pr2, _, _, _, _, rz = procesar_e_imprimir_regresion(Preg, 4, zonas, 2, 4, 1, 5)
    
    return Pr2


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



def leer_datos_excel(file_path):
    df = pd.read_excel(file_path)
    columna = pd.to_numeric(df['A'], errors='coerce').dropna()
    return columna


def obtener_siguiente_numero(columna):
    ultima_caida = columna.iloc[-1]
    return [columna[i + 1] for i in range(len(columna) - 1) if columna[i] == ultima_caida]


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
    grupo = columna.tail(20)
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
                distancia = 20
            else :
                distancia = 19
        apariciones[num] = distancia
    return apariciones


def ultima_jerarquia(columna):
    grupo = columna.tail(50)
    frecuencias = {num: grupo.tolist().count(num) for num in range(10)}  # Usa .count() en la lista
    #print(frecuencias)  # Ahora imprimirá las frecuencias correctamente
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
    
        
    """
    lista_sig.pop()
    Tot14=sum(columna[-11:])
    u30=sum(columna[-28:])/len(columna[-28:])
    u10=sum(columna[-20:])/len(columna[-20:])
    u6=sum(columna[-10:])/len(columna[-10:])
    u4=sum(columna[-3:])/len(columna[-3:])
    xxx=u6
    """
    
    # Calcular lista_30 y errores_30
    #print("Datos Bajos")
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
    u30=(u30-PTotal)/PTotal
    u10=(u10-PTotal)/PTotal
    u6=(u6-PTotal)/PTotal
    u4=(u4-PTotal)/PTotal

    #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}  {tam} \t\t Ultimos datos de 6:{u30:.3f}  {u10:.3f}   {u6:.3f}  {u4:.3f}",(225, 225, 225), (50, 250, 50)))
    return errores_30, errores_10, errores_6, errores_4, lista_sig, Tot14, PTotal, u30, u10, u6, u4, xxx


def procesar_e_imprimir_regresion(Pregunta, Pos, Lista, Nn, Tipo, start=0,stop=10):
    # Códigos ANSI
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[33;46m"
    L_30, L_15, L_6, L_4, L_sig, Sum14, PROM, u3, u1, u6, u4, ud = calcular_promedios_y_errores(Lista, Pos, Tipo)
    
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_4, L_sig)
    nuevo_dato = np.array([[u3, u1, u6, u4]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    Et=pd.Series(ETo[-80:])
    a1, a5, pp2, pp = primer_impres(Et, 2, -0.04, 0.0, 0.04)
    print("")
    extras=(a1, a5, pp2, pp)
    print(f"\033[31mGeneral :\033[0m", end="\t")
    print(f"\033[1;31;47m{PROM:.3f}\033[0m", end="\t\t") 
    
    P_gral=Pprom
    print(f"\x1b[48;5;157mRegresion:{Pprom:.3f} - {Pprom*0.95:.2f} - {Pprom*1.05:.2f}\033[0m", end="\t")
    #print(f"Otro: {Pprom*(1+ab):.3f}")
    print(" -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")
    n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral, start, stop, *extras)
    Valores=n_valores
    minY=min(n_valores)
    maxY=max(n_valores)
    """
    if P_gral<minY:  
        P_gral=minY*1.005
        
    if P_gral>maxY:
        P_gral=maxY*0.995
        
    n_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, P_gral, start, stop)
    
    """

    zz=(P_gral-minY)/(maxY-minY)
    print(f"\tNuevo Prom: {P_gral:.3f}\t\tRazon: {zz:.2f}")
    print("-----------------------          ------------------------------")
    Prom_Gral=P_gral

    if Pregunta==1:
        Prom_Gral = solicitar_nuevo_prom(minY, maxY)
        nuevos_valores, errores_ajustados, xx = imprimir_valores_y_errores(Sum14, Pos, Prom_Gral, start, stop)
        Valores=nuevos_valores
    
    return  errores_ajustados, Valores, Sum14, Prom_Gral, xx, zz
    


def imprimir_valores_y_errores(s14, Po, p_gral, start=0, stop=10, *extras):
    col_val="\033[93m"
    col_err="\033[96m"
    formateados = []
    errores     = []
    nuevos      = []
    ajustados   = []
    escalas     = []
    u           = []
    sinsigno    = []
    x=0
    
    for i in range(start, stop):
        val   = (s14 + i) / 17
        x=17
    
        val=float(val)    
        err   = (val - p_gral) / p_gral
        escala=error_to_scale(err)
        formateados.append(f"{val:.3f}\t")
        errores.append(f"{err:.3f}")
        escalas.append(f"{escala:.1f}")
        ajustados.append(err * -0.999 if err < 0 else err)
        nuevos.append(val)
        sinsigno.append(err)
        
    for v in escalas:  
        u.append(str(v))  
    
    linea5 = "\t".join(u)
    print(f"{col_val}{' '.join(formateados)}{COLORS['reset']}")
    print("\t".join(colorear2(float(e)) for e in errores))
    print(f"\033[95m{linea5}{COLORS['reset']}")
    
    uu=stop-start
    print(uu)
    
    if extras:
        a, b, c, d = extras
        if uu==4:
            pr1=sinsigno[0]
            pr2=sinsigno[1]
            pr3=sinsigno[2]
        
        elif uu==5:
            pr1=sinsigno[0]
            pr2=sinsigno[2]
            pr3=sinsigno[4]
        
        else:
            pr1=sinsigno[3]
            pr2=sinsigno[5]
            pr3=sinsigno[7]

        buscaprom(c, d, pr1, pr2, pr3, a, b)

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




def Imprimir_datos_listas(stats: dict, mode: int = 0):
    # ANSI settings
    BG_GRAY = "\x1b[48;2;220;220;220m"
    reset   = "\x1b[0m"
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
            f"{reset}"
        )
    
    def fmt_number(val: float) -> str:
        """ Sólo número (3 decimales), con fondo gris y texto rojo/azul. """
        r, g, b = (255, 0, 0) if val < 0 else (0, 0, 255)
        txt = f"{val:.2f}"

        return (
            f"{BG_GRAY}"
            f"\x1b[38;2;{r};{g};{b}m"
            f" {txt} "
            f"{reset}"
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
    reset   = "\x1b[0m"
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
            f"{reset}"
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
            f"{reset}"
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
    return f"{fg_code}{bg_code}{texto}{COLORS['reset']}"


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
    #azul  = "\033[34m"
    #rojo  = "\033[31m"
    amarillo = "\033[92m"

    num_color = COLORS['blue'] if valor >= 0 else COLORS['red']
    if dec==0:
        return f"{amarillo}{etiqueta}{COLORS['reset']} = {num_color}{valor:.0f}{COLORS['reset']}"
    else:
        if t==0:
            return f"{amarillo}{etiqueta}{COLORS['reset']} = {num_color}{valor:.3f}{COLORS['reset']}"
        else:
            return f"{amarillo}{etiqueta}{COLORS['reset']} = {num_color}{valor:.4f}{COLORS['reset']}"


def colorear2(valor):
    fondo = "\033[104m"  # Fondo gris claro
    if valor >= 0:
        texto = "\033[34m" if valor > 0 else "\033[30m"  # Azul si >0, negro si ==0
    else:
        texto = "\033[31m"  # Rojo si negativo
    return f"{fondo}{texto}{valor:.3f}{COLORS['reset']}"





def imprimir_tabla(Titulo, data, es_decimal=False, highlight_key=None):
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
            return f"{COLORS['red']}{cell}{COLORS['reset']}"
        if i in centro_idxs:
            return f"{COLORS['yellow']}{cell}{COLORS['reset']}"
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
        c=procesar_Histogramas( 2, 2, 30, Ss, F_dsig)
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
        #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}",(225, 225, 225), (117, 174, 90)))
        
        u30=(u30-pgral)/pgral
        u10=(u10-pgral)/pgral
        u6=(u6-pgral)/pgral
        u4=(u4-pgral)/pgral
    
    elif  zz < 5:
        
        lista_30 = medias_ventana(35)
        lista_10 = medias_ventana(26)
        lista_6  = medias_ventana(15)
        lista_4  = medias_ventana(4)
        lista_sig = [ sum(data[i-(nx-1): i+1]) / len(data[i-(nx-1): i+1]) for i in range(60, n+1) ]
        u30=sum(data[-34:])/len(data[-34:])
        u10=sum(data[-25:])/len(data[-25:])
        u6=sum(data[-14:])/len(data[-14:])
        u4=sum(data[-3:])/len(data[-3:])
        lista_sig.pop()
        #print(u6, tip)
        L_3, L_10, L_6, L_4, L_sig, bn, tam = last_interval_expand(lista_30, lista_10, lista_6, lista_4, lista_sig, u6, tip)
        best_svr, cv_score, PromT, ETo = aplicar_svr(L_3, L_10, L_6, L_4, L_sig)
        nuevo_dato = np.array([[u30, u10, u6, u4]])
        prediccion = best_svr.predict(nuevo_dato)
        Pprom=prediccion[0]
        #print(aviso_ansi(f"Nuevo dato: {Pprom:.3f}  {bn}",(225, 225, 225), (117, 174, 90)))
        
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

    def errores(lista_medias):
        return [ (m - pgral) / pgral for m in lista_medias ]

    errores_30 = errores(lista_30)
    errores_15 = errores(lista_10)
    errores_6  = errores(lista_6)
    errores_4  = errores(lista_4)
                      
    # 4) suma de los últimos 14 valores
    suma_14 = sum(data[-(nx-1):])
    
    return errores_30, errores_15, errores_6, errores_4, lista_sig, suma_14, pgral, u30, u10, u6, u4, Predm, tam


def calcular_nuevos_y_errores(Valores: Dict[str, float], Sum14: float, Prom_Gral: float, Nx, extra=None) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    errores_clave   = {}
    nuevos_clave    = {}
    formateados1    = []
    formateados2    = []
    Escalar         = []
    escalaprint     = []
    sinsigno        = []

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
        sinsigno.append(error)

    if extra is not None:
        a, b, c, d = extra
        pr1=sinsigno[2]
        pr2=sinsigno[5]
        pr3=sinsigno[8]
        buscaprom(c, d, pr1, pr2, pr3, a, b)
        print()
    return nuevos_clave, errores_clave, formateados1, formateados2, escalaprint


def procesar_regresion_Histo(Pr, zz, P, Lista, Valores, Nn, tip, start=1, stop=15 ):
    BG_BLUE = '\033[46m'
    RED_ON_YELLOW = "\033[37;45m"
    reset        = "\033[0m"
    y = dict(sorted(Valores.items(), key=lambda item: item[1]))
    Predm=0
    
    L_30, L_15, L_6, L_4, L_sig, Sum14, PROM, u3, u1, u6, u4, Predm, tam = promedios_y_errores_lista(Lista, zz, P, y, tip, Nn)
    best_svr, cv_score, PromT, ETo = aplicar_svr(L_30, L_15, L_6, L_4, L_sig)
    nuevo_dato = np.array([[u3, u1, u6, u4]])
    prediccion = best_svr.predict(nuevo_dato)
    Pprom=prediccion[0]
    er_30a, er_15a, er_6a, er_4a, lis_siga, df_debug = filtrar_segmentos_df(L_30, L_15, L_6, L_4, L_sig, u6)
    
    if len(df_debug)>30:
        best_svr1, cv_score1, mpe_all1, errors_all1 = aplicar_svr(er_30a, er_15a, er_6a, er_4a, lis_siga)
        pred = best_svr1.predict(nuevo_dato)
        Predm= pred[0]

    #resumen_200 = analizar_lista_pct(ETo, pct=200)
    Et=pd.Series(ETo[-80:])
    a1, a5, pp2, pp = primer_impres(Et, 2, -0.04, 0.0, 0.04)
    extras=(a1, a5, pp2, pp)
    print("\033[31mPromedio :\033[0m", end=" ")
    print(f"\033[1;91;47m{PROM:.3f}\033[0m", end="\t")  
    
    if len(L_30) < 70:
        print(f"No hay datos suficientes")
        # Ruta por defecto (cuando no hay datos suficientes)
        default = ({i: 0 for i in range(1, 10)}, None, 0)
        return default
    

    if (L_30 and L_15 and L_6 and L_sig and len(L_30) == len(L_15) == len(L_6) == len(L_sig) ):
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
        nuevos, errores, textoN, textoE, tt = calcular_nuevos_y_errores(y, Sum14, Pprom, Nn, extras)
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
       
        if Pprom<minY:
            Pprom=minY*1.004
        if Pprom>maxY:
            Pprom=maxY*0.996
        
        nuevos, errores, texto,textoE, tt = calcular_nuevos_y_errores(y, Sum14, Pprom, Nn)
        #linea3 = "\t".join(textoN)
        
        #linea4 = "\t".join(textoE)
        #print(f"\033[91m{linea3}\033[0m")
        #print(f"\033[97m{linea4}\033[0m")
        print("Nuevo Prom: ", "{:.2f}".format(Pprom))
        #print("--------       -----------       ---------        --------       ---------        --------")
        if Pr==1 :
            Prom_Gral = solicitar_nuevo_prom(minY, maxY)
            nuevos, errores, texto, textoE, tt = calcular_nuevos_y_errores(y, Sum14, Prom_Gral, Nn, extras)
            linea = "\t".join(textoN)
            linea1 = "\t".join(textoE)
            linea2 = "\t".join(tt)
            print(f"\033[92m{linea}\033[0m")
            print(f"\033[95m{linea1}\033[0m")
            print(f"\033[95m{linea2}\033[0m")
        return  errores, nuevos, Sum14

    else:
        print(f"Las listas no cumplen las condiciones")
        return None, None, None


def escalar_dic(d, escalar):
    return {k: v * escalar for k, v in d.items()}


def procesar_Histogramas(Preg, Zz:int, Pos: int, Nn, h_data, f_data: dict, tip, *proc_args):
    Y=len(h_data)
    default = ({i: 0 for i in range(1, 10)}, None, None)
        
    if Y > 120:
        Error_val, Nuevo_valor, Sum14 = procesar_regresion_Histo(Preg, Zz, Pos, h_data, f_data, Nn, tip, *proc_args)
        # 2) Ordenar frecuencias y errores
        if Sum14 !=0:
            caidas_ordenadas = ordenar_por_valor(f_data, ascendente=True)
            prom_ordenados   = ordenar_por_valor(Error_val, ascendente=True)
            llave_min = min(prom_ordenados, key=lambda k: abs(prom_ordenados[k])) #min(prom_ordenados, key=prom_ordenados.get)
            
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


def fmt(x, decimals=4):
    s = f"{x:.{decimals}f}"
    if s.startswith("0."):
        return s[1:]           # 0.1234 -> .1234
    if s.startswith("-0."):
        return "-" + s[2:]     # -0.1234 -> -.1234
    return s                  # otros casos (p.ej. 1.0000)


def primer_impres(lista, tip, x, y, z, *cae):
    last10 = pd.to_numeric(lista.tail(12), errors="coerce").dropna().astype(float)
    last6 = last10.iloc[-6:]
    def fmt(v, tip):
        v = float(v)
        if tip==0:
            return "{:.0f}".format(v) if v.is_integer() else "{:.1f}".format(v)
        else:
            return "{:.0f}".format(v) if v.is_integer() else "{:.3f}".format(v)
        
    vals = list(map(str, last10))

    def counts(s):
        mayor = int((s > y).sum())
        return mayor
    
    mayor10 = counts(last10)
    mayor6 = counts(last6)
    P20=lista.mean()
    P30=lista.tolist()
    Prome=sum(P30[-25:])/(len(P30[-25:]))
    promme=sum(P30[-4:])/(len(P30[-4:]))
    imprimir_grupos(lista[-12:], tip)
    if tip==0:
        print(COLORS['yellow'] + f"   M10= {mayor10}  M6= {mayor6}    PT {P20:.2f}    Prom {Prome:.2f}    Pr4 {promme:.2f}" + "\033[0m" + COLORS['reset'])
        printavgs(lista[-30:], 0)
    else:
        print(f"   M10= {mayor10}  M6= {mayor6}")
        print(COLORS['yellow'] + f"PT {P20:.3f}    Prom {Prome:.3f}    Pr4 {promme:.3f}" + COLORS['reset'])
    
    a1, a5, P2, P = resumen_triple(lista,  0)
    b1, b5, P2, P = resumen_triple(lista,  1)
    bb2=construir_jugadas(lista)
    #plot_jugadas(bb2, tip)


    if tip!=0:
        printavgs(lista[-30:], 1)

    print()

    imprime_estadisticas(lista, P20, Prome, x, y, z, tip)
    
    if cae:
        print(aviso_ansi(f"Este numero cayó :{cae}", (18, 35, 140), (255, 170, 170) ))

    print()
    return b1, b5, P2, P




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


def imprime_estadisticas(lista, Prom1, Prom2, X1, X2, X3, tipo):
    
    l3=[]
    l5=[]

    if tipo==0:
        er=[X1, X2, X3]
        a=valores_estadisticas(lista, 10, X1, X2, X3)
        l3.append(a)
        b=valores_estadisticas(lista, 8, X1, X2, X3)
        l3.append(b)
        l3.append(b)
        c=valores_estadisticas(lista, 4, X1, X2, X3)
        l3.append(c)
        y=sum(l3)/len(l3)
        print("P.T: ", "{:.2f}".format(y))
        
    elif tipo==1:
        #err=[-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        err=[-0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        a=valores_estadisticas5(lista, 20, err)
        l5.append(a)
        b=valores_estadisticas5(lista, 5, err)
        l5.append(b)
        #print()
        c=valores_estadisticas5(lista, 8, err)
        l5.append(c)
        #l5.append(c)
        d=valores_estadisticas5(lista, 4, err)
        l5.append(d)
        y=sum(l5)/len(l5)
        yy=(sum(l5)+c)/(len(l5)+1)
        print("P.T: ", "{:.2f}".format(y),"  ", "{:.2f}".format(yy) )

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
        yy=(sum(l5)+c)/(len(l5)+1)
        print("P.T: ", "{:.2f}".format(y),"  ", "{:.2f}".format(yy))
    
    if tipo==1:
        PromAxl(lista, Prom2, tipo)
    else:
        PromAxl(lista, Prom1, tipo)


def PromAxl(Lista, p, tipo):
    
    l=Lista.tolist()
    a=sum(l[-9:])
    a=p*10-a
    b=sum(l[-7:])
    b=p*8-b
    c=sum(l[-5:])
    c=p*6-c
    d=sum(l[-3:])
    d=p*4-d
    #e=sum(l[-3:])
    #e=p*4-e
    #print("tip", tipo, end="  ")
    
    if tipo>0:
        if (a>0.12):
            a=0.05
        if (b>0.12):
            b=0.05
        if (c>0.12):
            c=0.05
        if (d>0.12):
            d=0.05
        
        if (a<-0.12):
            a=-0.05
        if (b<-0.12):
            b=-0.05
        if (c<-0.12):
            c=-0.05
        if (d<-0.12):
            d=-0.05
            
    else:
        if a<0 :
            a=1
        if b<0 :
            b=1
        if c<0 :
            c=1
        if d<0 :
            d=1
        #if e<0 :
        #    e=1
    
    difs, prr = generar_diferencias(Lista, tipo)
    pr10=sum(difs[-10:])/len(difs[-10:])
    pr4=sum(difs[-4:])/len(difs[-4:])
    last10 = difs[-10:]
    #print("")
    
    # construir lista con coloreado en índice 1 (2º) e índice 5 (6º)
    formatted = []
    for i, x in enumerate(last10):
        if tipo>0:
            s = f"{x:.3f}"
        else:
            s = f"{x:.1f}"

        if i == 3:
            s = f"{COLORS['green']}{s}{COLORS['reset']}"
        elif i == 6:
            s = f"{COLORS['green']}{s}{COLORS['reset']}"
        formatted.append(s)

    if tipo>0:
        prome=(a+b+c+d)/4
        prome4=(a+b+c)/3
        prome3=(c+d)/2
        promem=(b+c+d)/3
        ptt=(prome + prome4 + prome3 + promem + promem)/5
        ptt1=error_to_scale(ptt)
        
        print(f"{a:.3f}  {b:.3f}  {c:.3f}  {d:.3f}      {{{ptt1:.1f}}} ")
        #print("{ " + ", ".join(formatted) + " }", end="")
    else :
        prome=(a+b+c+d)/4
        prome4=(b+c+d)/3
        prome3=(c+d)/2
        print(f"{a:.0f}  {b:.0f}  {c:.0f}  {d:.0f}         {prome:.0f}   {prome4:.0f}   {prome3:.0f}")
        

"""
def fun_promedios(lista, modo, t, h=3):
    FG_RED    = "\033[31m"
    FG_BLUE   = "\033[34m"
    BG_GREEN  = "\033[107m"
    FG_DEFAULT= "\033[39m"

    ultimos = lista[-100:]
    PromT = mean(ultimos)
    
    # Funciones auxiliares
    avg = lambda n: mean(lista[-n:])         # promedio de los últimos n elementos
    sum_n = lambda n: sum(lista[-n:])        # suma de los últimos n elementos
    # Cálculos principales
    a1, a5, Pp, P = 0.0, 0.0, 0.0, 0.0

    if h==3:
        nn=pd.Series(ultimos)
        n=nn.mean()
        a1, a5, Pp, P = primer_impres(nn, t, -0.04, 0, 0.04)
       
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
    
    
    return a1, a5, Pp, P
"""

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
    
    # construir filas de una tabla horizontal (header y fila de valores), aplicando mapa a índices originales si se da
    def build_table_rows(keys, vals, col_widths, index_map=None):
        header_cells = [k.center(w) for k, w in zip(keys, col_widths)]
        header = "  ".join(header_cells)
        val_cells = []
        for i, (v, w) in enumerate(zip(vals, col_widths)):
            text = v.center(w)
            orig_idx = index_map[i] if index_map is not None else i
            if orig_idx in green_indices:
                text = f"{COLORS['green']}{text}{COLORS['reset']}"
            elif orig_idx in yellow_indices:
                text = f"{COLORS['yellow']}{text}{COLORS['reset']}"
            elif orig_idx in red_indices:
                text = f"{COLORS['red']}{text}{COLORS['reset']}"
                
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
        ta=5
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


def resumen_triple(lista: List[float], t: int):
       
    def mean_of_slice(arr: List[float], start: int, length: int) -> float:
    
        if start < 0:
            start = 0
        if start >= len(arr):
            return float('nan')
        
        slice_ = arr[start:start+length]
        
        return sum(slice_) / len(slice_) if len(slice_) > 0 else float('nan')
    
    n = len(lista)
    if t==0:
        win_long, win_short = 10, 5
    else:
        win_long, win_short = 20, 10

    # Índices de interés
    idx_antep = n - 4
    idx_pen   = n - 3
    idx_last  = n - 2

    if t==0:
        idx  = n - 10
        idx5 = n - 5
    else:
        idx  = n - 20
        idx5 = n - 10

    def fmt(x):
        if x != x:  # NaN
            return "0.0"
        return f"{x:.3f}"

    def bloque_para(pos: int) -> Tuple[str,str,str,float,float]:
        inicio = max(0, pos - 39)   # 40 valores hacia atrás
        fin = pos + 1               # incluir el valor en pos
        ult40 = lista[inicio:fin]

        a = sum(ult40) / len(ult40) if len(ult40) > 0 else float('nan')
        b = mean_of_slice(lista, (pos - (win_long - 1)) + 1, win_long)
        s = mean_of_slice(lista, pos - (win_short -1)+1, win_short)
        return fmt(a), fmt(b), fmt(s), b, s

    # Promedios dinámicos
    a1, b1, s1, _,_ = bloque_para(idx_antep)
    a2, b2, s2, _,_ = bloque_para(idx_pen)
    a3, b3, s3, P2, P = bloque_para(idx_last)

    # valores individuales
    valor_pen = lista[idx_pen+1] if 0 <= idx_pen < n else 0
    valor_last = lista[idx_last+1] if 0 <= idx_last < n else 0
    valor_22 = lista[idx-2] if 0 <= idx_last < n else 0
    valor_21 = lista[idx-1] if 0 <= idx_last < n else 0
    valor_20 = lista[idx] if 0 <= idx_last < n else 0
    valor_7 = lista[idx5-2] if 0 <= idx_last < n else 0
    valor_6 = lista[idx5-1] if 0 <= idx_last < n else 0
    valor_5 = lista[idx5] if 0 <= idx_last < n else 0

    # Formateo
    str_pen = formatear_valor_t(valor_pen)
    str_last = formatear_valor_t(valor_last)
    str_22c = colored(formatear_valor_t(valor_22), "cyan")
    str_21c = colored(formatear_valor_t(valor_21), "cyan")
    str_20c = colored(formatear_valor_t(valor_20), "cyan")
    str_7c  = colored(formatear_valor_t(valor_7), "blue")
    str_6c  = colored(formatear_valor_t(valor_6), "blue")
    str_5c  = colored(formatear_valor_t(valor_5), "blue")    

    # Bloques con prom40 en bloque1
    bloque1 = f"{{¦{a1}, {b1}, {s1}¦}}{{{str_pen}}}"
    bloque2 = f"{{¦{a2}, {b2}, {s2}¦}}{{{str_last}}}"
    bloque3 = f"{{{str_20c}¦{a3}, {b3}, {s3}¦{str_5c}}}"

    print(f"{bloque1}  {bloque2}  {bloque3}")
    return valor_20, valor_5, P2, P


def colored(text: str, color: str) -> str:
    code = COLORS.get(color, "")
    #if not code or not sys.stdout.isatty():
    #    return text
    return f"{code}{text}{COLORS['reset']}"


def formatear_valor_t(v, leading_zero: bool = True):
    # Nombre auxiliar para formatear floats con 3 decimales y posible supresión del 0
    def _fmt_float_three(x: float) -> str:
        s = f"{x:.3f}"  # e.g. "0.012" or "-0.012"
        if leading_zero:
            return s
        # quitar el '0' antes del punto en positivos y el '0' tras el signo en negativos
        if s.startswith("-0."):
            return "-." + s[3:]
        if s.startswith("0."):
            return "." + s[2:]
        return s

    # None explícito -> "0" (modifica si prefieres "0.000" u otro)
    if v is None:
        return "0"

    # int nativo (excluir bool)
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:.0f}"

    # float nativo
    if isinstance(v, float):
        if v.is_integer():
            return f"{v:.0f}"
        return _fmt_float_three(v)

    # intentar convertir a float (cubre strings y numpy scalars)
    try:
        vf = float(v)
    except Exception:
        return str(v)

    # tras conversión decidir formato
    if math.isfinite(vf) and vf.is_integer():
        return f"{vf:.0f}"
    return _fmt_float_three(vf)


def imprimir_grupos(datos, t, grupo=4) -> None:

    def formatea(x: float) -> str:
        if t == 0:
            # redondea y muestra sin decimales
            return f"{x:.0f}"
        # float: 3 decimales
        return f"{x:.3f}"

    n = len(datos)
    grupos = []
    for i in range(0, n, grupo):
        piezas = []
        for x in datos[i:i+grupo]:
            texto = formatea(x)
            if x < 0:
                piezas.append(f"{COLORS['red']}{texto}{COLORS['reset']}")
            elif x > 0:
                piezas.append(f"{COLORS['blue']}{texto}{COLORS['reset']}")
            else:
                piezas.append(texto)
        grupos.append("  ".join(piezas))
    print("    ".join(grupos), end="  ")



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

    if t>0:
        fmt = lambda v: f"{v:.3f}"
    else:
        fmt = lambda v: f"{v:.2f}"

    s1 = fmt(valor1)
    s2 = f"{COLORS['red']}{fmt(valor2)}{COLORS['reset']}"
    s3 = fmt(valor3)
    print(f"{{{s1} {s2} {s3}}}")


def printavgs(data, t):
    n = len(data)

    def window_avg_abs(end_offset, window_size):
        # end_offset: 5 => ventana termina en índice n-5
        end_pos = n - end_offset
        start_pos = end_pos - window_size + 1
        window = data[start_pos:end_pos + 1]
        return sum(window) / window_size

    avgs_10 = [window_avg_abs(off, 20) for off in range(10, 0, -1)]
    avgs_5  = [window_avg_abs(off, 10)  for off in range(10, 0, -1)]
    avg_of_avgs_10 = sum(avgs_10) / len(avgs_10)
    avg_4_10 = sum(avgs_10[-5:]) / len(avgs_10[-5:])
    avg_of_avgs_5 = sum(avgs_5) / len(avgs_5)
    avg_4_5 = sum(avgs_5[-5:]) / len(avgs_5[-5:])

    if t == 0:
        left = "  ".join(f"{v:.1f}" for v in avgs_10)
        right = "  ".join(f"{v:.1f}" for v in avgs_5)
        left += "    " + COLORS['cyan'] + f"{avg_of_avgs_10:.1f}" + COLORS['magenta']
        right += "    " + COLORS['cyan'] + f"{avg_of_avgs_5:.1f}" + COLORS['magenta']
        left += "  " + COLORS['yellow'] + f"{avg_4_10:.1f}" + COLORS['magenta']
        right += "  " + COLORS['yellow'] + f"{avg_4_5:.1f}" + COLORS['magenta']
    
    else:
        left = "  ".join(formatear_valor_t(v, leading_zero=False) for v in avgs_10[-6:])
        right = "  ".join(formatear_valor_t(v, leading_zero=False) for v in avgs_5[-6:])
        left += "   " + COLORS['cyan'] + formatear_valor_t(avg_of_avgs_10, leading_zero=False) + COLORS['magenta']
        right += "   " + COLORS['cyan'] + formatear_valor_t(avg_of_avgs_5, leading_zero=False) + COLORS['magenta']
        left += "  " + COLORS['yellow'] + formatear_valor_t(avg_4_10, leading_zero=False) + COLORS['magenta']
        right += "  " + COLORS['yellow'] + formatear_valor_t(avg_4_5, leading_zero=False) + COLORS['magenta']


    print(COLORS['magenta'] + f"{left}\t  {right}" +  COLORS['reset'])


def buscaprom(Pp2, Pp, r1, r2, r3, a1, a5):
    x1 = (Pp2 * 20 - a1 + r1)/20.0
    y1 = (Pp2 * 20 - a1 + r2)/20.0
    z1 = (Pp2 * 20 - a1 + r3)/20.0
    x2 = (Pp * 10 - a5 + r1)/10.0
    y2 = (Pp * 10 - a5 + r2)/10.0
    z2 = (Pp * 10 - a5 + r3)/10.0

    print(f"{{{x1:.3f} * {y1:.3f} * {z1:.3f}}}  {{{x2:.3f} * {y2:.3f} * {z2:.3f}}}")


def construir_jugadas(datos, n_tuplas=4, ventanas=(80, 35, 20, 10)):
    datos = list(map(float, datos))
    n = len(datos)
    if n < max(ventanas):
        raise ValueError(f"Se requieren al menos {max(ventanas)} datos para aplicar la lógica.")

    jugadas = []
    # Recorremos desde el 6º antes del final hacia atrás
    for k in range(n_tuplas):
        idx_resp = n - (4 - k)   # 5º antes del final, luego 4º, ..., hasta el último
        if idx_resp < 0 or idx_resp >= n:
            break

        corte = idx_resp

        # Promedios dinámicos según ventanas
        promedios = []
        for w in ventanas:
            # Aseguramos que siempre haya datos suficientes
            inicio = max(0, corte - w)
            if inicio < corte:
                promedios.append(np.mean(datos[inicio:corte]))
            else:
                promedios.append(float('nan'))  # marcador si no hay datos

        resp = datos[idx_resp]

        # Tupla de 5 valores (4 promedios + respuesta)
        jugadas.append(tuple(promedios + [resp]))

    return jugadas


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


def zonitas(Nume, k):
    print(aviso_ansi(f"Resultados para Numeros Zonas:",(118, 5, 35), (240, 225, 100)))
    zonifica = lambda x: 1 if x < 3 else 2 if x < 5 else 3 if x < 7 else 4
    for v in range(k, 0, -1):
        Numeros=Nume[:-v]
        caida = Nume.iloc[-v]
        cae=zonifica(caida)
        yy=Zonas_Series(Numeros, 0, cae)
        print("")
    #print(aviso_ansi(f"Resultados para Numeros Zonas:",(118, 5, 30), (240, 220, 90)))
    a=Zonas_Series(Nume, 1)
    print(" ".join(f"{v:.3f} " for v in a))
    print("\x1b[48;5;71m  = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   \x1b[0m")
    print("")
    return a


def numemedios(Nume, k):
    print(aviso_ansi(f"Resultados para Numeros Medios:",(118, 5, 35), (240, 225, 100)))
    Nume2=Lista2_series(Nume)
    for v in range(k, 0, -1):
        Numeros=Nume2[:-v]
        caida = Nume2.iloc[-v]
        a1, a5, Prr, P = primer_impres(Numeros, 0, 1, 3, 5, caida)
        print("")
        Pr_Num, _, _, _, _, rz = procesar_e_imprimir_regresion(0, 4, Numeros, 2, 0,  1, 6)

    #print(aviso_ansi(f"Resultados para Numeros Medios:",(118, 5, 30), (240, 220, 90)))
    a1, a5, Pp, P = primer_impres(Nume2, 0, 1, 3, 5)
    b, _, _, _, _, rz = procesar_e_imprimir_regresion(1, 4, Nume2, 2, 0,  1, 6)
    print(" ".join(f"{v:.3f} " for v in b))
    print("\x1b[48;5;71m= = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   = = =   \x1b[0m")
    print("")
    return b


def numerotes(nume, k):
    print(aviso_ansi(f"Resultados para Numeros Completos 0-9:",(118, 5, 35), (240, 225, 100)))
    for v in range(k, 0, -1):
        Numeros=nume[:-v]
        caida = nume.iloc[-v]
        a1, a5, Pp, P = primer_impres(Numeros, 0, 2, 4, 7, caida)
        Pr_Num, _, _, _, _, rz = procesar_e_imprimir_regresion(0, 4, Numeros, 2, 0)
        #buscaprom(Pp, Pr_Num[3], Pr_Num[6], a1, a5)
    
    #print(aviso_ansi(f"Resultados para Numeros Completos 0-9:",(118, 5, 30), (240, 220, 90)))
    a1, a5, Pp, P = primer_impres(nume, 0, 2, 4, 7)
    c, _, _, _, _, rz = procesar_e_imprimir_regresion(1, 4, nume, 2, 0)
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
    print(aviso_ansi(f"Resultados para Frecuencias por Zona:",(220, 50, 130), (125, 220, 210)))
    for v in range(k, 0, -1):
        Nume=numero[:-v]
        Numeros=pd.Series(Nume)
        caida = numero[-v]
        F_datos=Semanas(Numeros)
        fg=zones_by_freq(Numeros, 0, caida)
    print(aviso_ansi(f"Resultados para Frecuencias por Zona:",(220, 50, 130), (125, 220, 210)))
    Numeros=pd.Series(numero)
    F_datos=Semanas(Numeros)
    fg=zones_by_freq(Numeros, 1)
    
    return fg


def ciclos(numero, j):
    jerarquia, Posi = calcular_jerarquias(numero)
    a=zonaciclos(Posi, j)
    print(aviso_ansi("Frecuencias en 50 jugadas  :", (220, 50, 130), (125, 220, 210)))
    for v in range(j, 0, -1):
        
        Numeros=numero[:-v]
        Ultima_Jerarquia=ultima_jerarquia(Numeros)
        jerarquias, Posic = calcular_jerarquias(Numeros)
        F_datos=Semanas(Numeros)
        claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_datos[k]))
        caida = Posi[-v]
        N10 = Posic[-90:]
        nn=pd.Series(N10)
        a1, a5, Pp, P = primer_impres(nn, 0, 2, 5, 8, caida)
        ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
        
        Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xxx, rz = procesar_e_imprimir_regresion(0, 4, Posic, 2, 2, 1, 11)
        print("\x1b[38;5;71m======================================================================================\x1b[0m")
        print()
    print(aviso_ansi("Frecuencias en 50 jugadas  :", (220, 50, 130), (125, 220, 210)))
    Ultima_Jerarquia=ultima_jerarquia(numero)
    jerarquias, Posic = calcular_jerarquias(numero)
    F_datos=Semanas(numero)
    claves_ordenadas = sorted(Ultima_Jerarquia.keys(), key=lambda k: (Ultima_Jerarquia[k], -F_datos[k]))
    ranking_dict = {rank: clave for rank, clave in enumerate(claves_ordenadas[:10])}
    N10 = Posic[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 2, 5, 8)
    Pr_Pos_val, Pr_Pos_err, Sum14, PromGral, xxx, rz = procesar_e_imprimir_regresion(1, 4, Posic, 2, 2, 1, 11)
    print("\x1b[38;5;71m======================================================================================\x1b[0m")
    #print()
        
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

    print(aviso_ansi("\nOrden de jerarquías :", (220, 50, 130), (125, 220, 210)))
    print("Id\t\t" + "\t".join(str(k) for k in claves_ordenadas))
    print("Repet\t\t" + "\t".join(str(Ultima_Jerarquia[k]) for k in claves_ordenadas))
    print("Apar\t\t" + "\t".join(str(F_datos[k]) for k in claves_ordenadas))
    print("\x1b[38;5;71m===========================================================================================\x1b[0m")
    print("")

    return errores_dict,sorted_keys


def numero_final(nu, err, sorted):
    # Calcula los mínimos
    min_jer = min(err.values())
    min_num = min(nu)

    ErrorNUm={}
    for k in sorted:
        jer = err[k]
        num = nu[k]
        ErrorNUm[k] = (abs(3*num) + abs(2*jer)) / 5 
        
        # Formatea cada celda, poniendo rojo si coincide con el mínimo
        s_jer = f"{jer:.3f}"
        if jer == min_jer:
            s_jer = f"{COLORS['red']}{s_jer}{COLORS['reset']}"
        
        s_num = f"{num:.3f}"
        if num == min_num:
            s_num = f"{COLORS['red']}{s_num}{COLORS['reset']}"
        #print(f"{k}\t{s_jer}\t\t")
        #print(f"{k}\t{s_jer}\t\t{s_num}\t\t{ErrorNUm[k]:.3f}")

    ErrorOrdenado=ordenar_por_valor(ErrorNUm, ascendente=True)
    #print()
    imprimir_tabla("\nErrores Promedios Numeros ", ErrorOrdenado, es_decimal=True)
    return ErrorOrdenado, ErrorNUm


def Histog3(nume, k, Max):
    for v in range(k, 0, -1):
        Histog=obtener_historial_caidas(nume, Max)
        print_Histo(Histog)
        caida = Histog[-v]
        Numeros=Histog[:-v]
        F_datos=Semanas(nume[:-v])
        #print(F_datos)
        H3 = Histo2_con_map(Numeros, 3)
        F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_datos.items()}
        cae=(caida - 1) // 3 + 1
        N10=H3[-90:]
        nn=pd.Series(N10)
        a1, a5, Pp, P = primer_impres(nn, 0, 2, 3, 4, cae) 
        a2=procesar_Histogramas(0, 3, 1, 27, H3, F_datos_3, 6)
        print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        print("")
        #print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    
    Numeros=obtener_historial_caidas(nume, Max)
    #print_Histo(Numeros)
    F_datos=Semanas(nume)
    #print(F_datos)
    H3 = Histo2_con_map(Numeros, 3)
    F_datos_3 = {k: (v - 1) // 3 + 1 for k, v in F_datos.items()}
    N10=H3[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 2, 3, 4)
    print("")
    a=procesar_Histogramas(1, 3, 1, 27, H3, F_datos_3, 6)
    #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
    print(" ".join(f"{v:.3f} " for k, v in a.items()))
    print("\x1b[38;5;24m=============================================================================================================\x1b[0m")
    print()
    return a


def Histog2(Nume, k, Max):
    for v in range(k, 0, -1):
        Histog=obtener_historial_caidas(Nume, Max)
        caida = Histog[-v]
        Numeros=Histog[:-v]
        N10=Numeros[-90:]
        nn=pd.Series(N10)
        #a1, a5, Pp, P = primer_impres(nn, 0, 3, 7, 11, caida)
        #print("")
        F_datos=Semanas(Nume[:-v])
        #print(F_datos)
        H2=Histo2_con_map(Numeros, 2)
        F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_datos.items()}
        N10=H2[-90:]
        nn=pd.Series(N10)
        a1, a5, Pp, P = primer_impres(nn, 0, 2, 4, 6)
        print("") 
        b=procesar_Histogramas(0, 3, 1, 27, H2, F_datos_2, 5)
        #print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
        print(" ".join(f"{v:.3f} " for k, v in b.items()))
        print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
        print("")

    Numeros=obtener_historial_caidas(Nume, Max)
    F_datos=Semanas(Nume)
    #print(F_datos)
    H2 = Histo2_con_map(Numeros, 2)
    F_datos_2 = {k: (v - 1) // 2 + 1 for k, v in F_datos.items()}
    N10=H2[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 2, 3, 4)
    print("")
    a=procesar_Histogramas(1, 3, 1, 27, H2, F_datos_2, 5)
    print("---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---  ---")
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
        N10=Numeros[-90:]
        nn=pd.Series(N10)
        a1, a5, Pp, P = primer_impres(nn, 0, 4, 8, 12, caida)
        print("")
        F_datos=Semanas(Nume[:-v])
        #print_Histo(Numeros)
        e1=procesar_Histogramas(0, 3, 2, 27, Numeros, F_datos, 3)
        print("\x1b[38;5;71m=============================================================================================================\x1b[0m")
        print("")
        
    print_recencias(Nume, k=3)
    Numeros=obtener_historial_caidas(Nume, maximo)
    N10=Numeros[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 4, 8, 12)
    print("")
    F_datos=Semanas(Nume)
    a=procesar_Histogramas(1, 3, 1, 27, Numeros, F_datos, 3)
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
    print(aviso_ansi("Posicion de Caidas:", (118, 5, 30), (240, 220, 90) ))
    numero=jerarquia_histo(nume)
    zonifica = lambda x: 1 if x < 4 else 2 if x < 6 else 3 if x < 8 else 4
    lis=[zonifica(x) for x in numero]
    #Imprime_historial(lis, 1, 1, 5)
    print(aviso_ansi("Posibilidades Zonales de Caidas:", (118, 5, 30), (240, 220, 90) ))
    for v in range(k, 0, -1):
        Numeros=numero[:-v]
        caida = numero[-v]
        lis=[zonifica(x) for x in Numeros]
        N10 = lis[-90:]
        nn=pd.Series(N10)
        cai=zonifica(caida)
        a1, a5, Pp, P = primer_impres(nn, 0, 1, 2, 3, cai)
        print("")
        yY, _, _, _, _, rz = procesar_e_imprimir_regresion(0, 4, lis, 0, 0, 1, 5)
        print("")
    print(aviso_ansi("Posibilidades Zonales de Caidas:", (118, 5, 30), (240, 220, 90) ))
    lis=[zonifica(x) for x in numero]
    N10 = lis[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 1, 2, 3)
    print("")
    a, _, _, _, _, rz = procesar_e_imprimir_regresion(1, 4, lis, 0, 0, 1, 5)
    print(" ".join(f"{v:.3f} " for  v in a))
    print("")

    print(aviso_ansi("Posibilidades por Puesto Semanas:", (118, 5, 30), (240, 220, 90) ))
    for v in range(k, 0, -1):
        Numeros=numero[:-v]
        caida = numero[-v]
        
        N10 = Numeros[-90:]
        nn=pd.Series(N10)
        a1, a5, Pp, P = primer_impres(nn, 0, 2, 5, 8, caida)
        print("")
        Pb_His2, _, _, _, _, rz = procesar_e_imprimir_regresion(0, 4, Numeros, 0, 2, 1, 11)
        print(" ".join(f"{v:.3f} " for  v in Pb_His2))
        print("")
    
    print(aviso_ansi("Posibilidades por Puesto Semanas:", (118, 5, 30), (240, 220, 90) ))
    N10 = numero[-90:]
    nn=pd.Series(N10)
    a1, a5, Pp, P = primer_impres(nn, 0, 2, 5, 8)
    print("")
    b, _, _, _, _, rz = procesar_e_imprimir_regresion(1, 4, numero, 0, 2, 1, 11)
    print(" ".join(f"{v:.3f} " for  v in b))
    print("\x1b[38;5;71m====     ====     ====     ====     ====     ====     ====     ====     ====     ====\x1b[0m")
    print("")
    res = [b[i] + a[zonifica(i + 1) - 1] for i in range(len(b))]

    #at, a1, a2, a3, a4, a5 =mascaidosp(numero, 225, 175, 125, 75, 25, 1, 11)
    #imprimir_diccionarios_tabla([a1, a2, a3, a4, a5], width=8, precision=3)
    return res



COLORS = {
        "reset": "\033[0m",
        "red":   "\033[31m",
        "green": "\033[32m",
        "yellow":"\033[33m",
        "blue":  "\033[34m",
        "magenta":"\033[35m",
        "cyan":  "\033[36m",
        "bold":  "\033[1m",
    }    


def main(file_path):
    #Lotery(COLORS["red"], COLORS["reset"])
    print(aviso_ansi("\nE M P E Z A M O S.....", (20, 210, 100), (215, 120, 120) ))
    Inic = datetime.datetime.now().strftime("%H:%M:%S")
    veces=0
    Maximo=20
    Numero = leer_datos_excel(file_path)
    print(aviso_ansi("\t\t\t\tEmpezando con Números :", (118, 5, 30), (240, 220, 120) ))
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
    #imprimir_tabla("Probabilidad Caidas Histograma ", H_histo, es_decimal=True)
    #print("")
    imprime_dict_color_horizontal_both(numbers)
    print("Pruebas.....")
    imprime_dict_color_horizontal_both(numb3rs_d)
    imprime_dict_color_horizontal_both(errores_dict)
    imprime_dict_color_horizontal_both(cai)
    imprime_dict_color_horizontal_both(H_histo)
    
    imprimir_tabla("Probabilidad Caidas Histograma ", Ordenados, es_decimal=True)
    print("")
    print(aviso_ansi("\nTERMINAMOS HISTOGRAMAS \n", (220, 110, 10), (120, 220, 200) ))
    print("")
    print("  --  Pseudo probabilidad jerárquica  --  ")
    print("")
    Prior1=calcular_alpha_prior(Numero)
    Prior=ordenar_por_valor(Prior1, ascendente=False)
    
    #Probab_mayor = aplicar_regresion_logistica_mayor_menor(Numero)
    #if Probab_mayor is not None:
    #    print(f"\nProbabilidad (Reg. Logística) de que el siguiente número sea mayor que 4: {Probab_mayor:.4f}")

    #Probab_par = aplicar_regresion_logistica_par_impar(Numero)
    #if Probab_par is not None:
    #    print(f"Probabilidad (Reg. Logística) de que el siguiente número sea par: {Probab_par:.4f}")
    print()

    """
    imprimir_tabla("Probabilidad de Numeros ", numbers, es_decimal=True)
    print("")
    imprimir_tabla("Posicion de Caida ", Cai, es_decimal=True)
    imprimir_tabla("Probabilidad de Histograma ", H_histo, es_decimal=True)
    print(aviso_ansi(f"\nTERMINAMOS AQUI : ", (118, 5, 30), (240, 220, 100) ))
    
       
    print("  --  Probabilidades recencia números  siguientes --  ")
    print("")
    #HIs_Sig=LLama_Sigui(Numeros)
    """
    print(Inic)
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    #usuario = input("Por favor, introduce tu nombre: ")
    

if __name__ == "__main__":
    print("Hello World")
    file_path = 'D:/loter.xlsx'
    main(file_path)