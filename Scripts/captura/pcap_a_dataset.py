#!/usr/bin/env python3
"""
Extractor DNS compatible con Mendeley DOI 10.17632/c4n7fckkz3.3
Genera 22 columnas: user_ip, domain, timestamp, attack, request, +18 features
Uso: python pcap_a_dataset.py --from-capturas --label 1/0
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scapy.all import rdpcap, DNS, DNSQR, IP, IPv6
import tldextract

OUTPUT_DIR   = Path("/home/quasar/Documents/TFG/Datasets")
CAPTURAS_DIR = Path("/home/quasar/Documents/TFG/capturas")
WORDS_PATH = Path("/home/quasar/Documents/TFG/mendeley/english_words.txt")

QTYPE_NAMES: Dict[int, str] = {
    1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 10: "NULL",
    12: "PTR", 15: "MX", 16: "TXT", 28: "AAAA", 33: "SRV", 255: "ANY",
}
TYPE_TO_QTYPE: Dict[str, int] = {"A": 1, "NULL": 10, "TXT": 16}


DEFAULT_WINDOW_SIZE = 10
MIN_WORD_LEN        = 3   # igual que en Mendeley (palabras de ≥ 3 chars)

CSV_COLUMN_NAMES = [
    "user_ip", "domain", "timestamp", "attack", "request",
    "len", "subdomains_count", "w_count", "w_max", "entropy",
    "w_max_ratio", "w_count_ratio", "digits_ratio", "uppercase_ratio",
    "time_avg", "time_stdev", "size_avg", "size_stdev", "throughput",
    "unique", "entropy_avg", "entropy_stdev",
]

# Filtro de corrupción (bytes > 127 en QNAME)
def _qname_has_nonascii(raw: bytes) -> bool:
    return any(b > 127 for b in raw)

def _fqdn_is_clean(fqdn: str) -> bool:
    """True si todos los caracteres son ASCII imprimible (0x20-0x7E).
    No descarta por longitud; los dominios largos de iodine son válidos.
    """
    return all(0x20 <= ord(c) <= 0x7E for c in fqdn)

# Léxico de palabras en inglés
def load_english_words() -> Optional[set]:
    """Carga el léxico de palabras inglesas desde WORDS_PATH.

    El fichero de Mendeley tiene una palabra por línea con posibles \\r\\n.
    Se descartan palabras de longitud < MIN_WORD_LEN (< 3 chars).
    Si el fichero no existe, las features w_* serán 0 (con aviso).
    """
    if not WORDS_PATH.exists():
        print(
            f"No existe {WORDS_PATH}.\n"
            "Las features w_count, w_max, w_max_ratio, w_count_ratio seran 0.\n",
            file=sys.stderr,
        )
        return None

    words: set = set()
    with open(WORDS_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()   # elimina \r\n y espacios
            if len(w) >= MIN_WORD_LEN:
                words.add(w)

    print(f"Lexico: {len(words):,} palabras (>= {MIN_WORD_LEN} chars) cargadas desde {WORDS_PATH.name}")
    return words

# Extracción de TLD y subdominio
def split_fqdn(fqdn: str) -> Tuple[str, str, str]:
    """Devuelve (subdomains, registered_domain, tld).
    Fallback simple para TLDs no estándar (.local, .lab, .tfg, etc.).
    """
    fqdn = fqdn.rstrip(".").lower()
    ext = tldextract.extract(fqdn)
    if ext.suffix and ext.domain:
        return ext.subdomain, ext.domain, ext.suffix
    parts = fqdn.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-2]), parts[-2], parts[-1]
    elif len(parts) == 2:
        return "", parts[0], parts[1]
    else:
        return "", fqdn, ""

def get_registered_domain(fqdn: str) -> str:
    _, reg, tld = split_fqdn(fqdn)
    return f"{reg}.{tld}" if tld else reg


# Features individuales (sin estado)
def shannon_entropy(s: str) -> float:
    """Entropía de Shannon del string s (en bits).
    Se calcula sobre todos los caracteres incluyendo puntos.
    """
    if not s:
        return 0.0
    n = len(s)
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def find_english_words(text: str, word_set: set) -> List[str]:
    """Busca TODAS las substrings de text que estén en word_set.

    Devuelve lista con todas las ocurrencias (incluyendo solapamientos).
    w_count = len(resultado)
    """
    text_l = text.lower()
    n = len(text_l)
    found: List[str] = []
    for i in range(n):
        for j in range(i + MIN_WORD_LEN, min(i + 64, n) + 1):
            sub = text_l[i:j]
            if sub in word_set:
                found.append(sub)
    return found


def compute_covered_chars(text: str, words: List[str]) -> int:
    """Cuenta las posiciones únicas del texto cubiertas por palabras encontradas.

    Busca todas las ocurrencias de cada palabra única y usa un conjunto para
    no duplicar posiciones solapadas.

    w_count_ratio = covered_chars / base_len
    """
    if not words:
        return 0
    text_l    = text.lower()
    covered   = set()
    seen      = set()

    for w in words:
        if w in seen:
            continue
        seen.add(w)
        start = 0
        while True:
            idx = text_l.find(w, start)
            if idx == -1:
                break
            covered.update(range(idx, idx + len(w)))
            start = idx + 1  # avanza 1 para encontrar solapamientos

    return len(covered)


def individual_features(fqdn: str, word_set: Optional[set]) -> dict:
    """Calcula las 9 features individuales (stateless) de una query DNS.

    Definiciones según dataset_description.txt de Mendeley:
   
    len              : longitud total del FQDN (sin punto final, en minúsculas)
    subdomains_count : número de etiquetas en la parte de subdominio
    w_count          : número de palabras inglesas encontradas como substrings
                       en el subdominio (sin puntos), incluye solapamientos
    w_max            : longitud de la palabra inglesa más larga encontrada
    entropy          : entropía de Shannon del FQDN completo (bits)
    w_max_ratio      : w_max / len(fqdn_sin_puntos)
    w_count_ratio    : caracteres_cubiertos_por_palabras / len(fqdn_sin_puntos)
                       (unión de posiciones, sin contar solapamientos dos veces)
    digits_ratio     : fracción de caracteres dígito en el FQDN (sin puntos)
    uppercase_ratio  : fracción de caracteres en mayúscula en el FQDN (sin puntos)
    """
    fqdn_clean = fqdn.rstrip(".").lower()
    sub, _, _  = split_fqdn(fqdn)

    # Longitud total del FQDN
    total_len = len(fqdn_clean)

    # Número de etiquetas de subdominio 
    subdomains_count = len(sub.split(".")) if sub else 0

    # Base para los ratios, FQDN completo sin puntos
    fqdn_no_dots = fqdn_clean.replace(".", "")
    base_len     = len(fqdn_no_dots) if fqdn_no_dots else 1

    # Subdominio sin puntos 
    sub_no_dots = sub.replace(".", "")

    # Entropía de Shannon del FQDN completo 
    entropy = shannon_entropy(fqdn_clean)

    # Ratios de dígitos y mayúsculas (sobre FQDN original sin puntos) 
    fqdn_orig_no_dots = fqdn.rstrip(".").replace(".", "")
    orig_len          = len(fqdn_orig_no_dots) if fqdn_orig_no_dots else 1

    digits_ratio    = sum(c.isdigit() for c in fqdn_no_dots)    / base_len
    uppercase_ratio = sum(c.isupper() for c in fqdn_orig_no_dots) / orig_len

    # Features de palabras inglesas 
    if word_set is not None and sub_no_dots:
        words   = find_english_words(sub_no_dots, word_set)
        w_count = len(words)
        w_max   = max((len(w) for w in words), default=0)
        covered = compute_covered_chars(sub_no_dots, words)
    else:
        words   = []
        w_count = 0
        w_max   = 0
        covered = 0

    return {
        "len":              total_len,
        "subdomains_count": subdomains_count,
        "w_count":          w_count,
        "w_max":            w_max,
        "entropy":          entropy,
        "w_max_ratio":      w_max / base_len,
        "w_count_ratio":    covered / base_len,
        "digits_ratio":     digits_ratio,
        "uppercase_ratio":  uppercase_ratio,
    }


# Extracción de queries DNS desde el pcap
def extract_dns_queries(pcap_path: str) -> List[dict]:
    """Extrae queries DNS con Scapy.

    Filtra bytes > 127 (corruptos / Base128) pero NO por longitud del dominio.
    Los dominios largos en Base32/Base64u son ASCII limpio y pasan el filtro.
    """

    records: List[dict] = []
    filtered = 0

    try:
        pkts = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error leyendo {pcap_path}: {e}", file=sys.stderr)
        return records

    for pkt in pkts:
        if not pkt.haslayer(DNS):
            continue
        dns = pkt[DNS]
        if dns.qr != 0 or dns.qdcount < 1:   # solo queries (QR=0)
            continue
        if pkt.haslayer(IP):
            src_ip = pkt[IP].src
        elif pkt.haslayer(IPv6):
            src_ip = pkt[IPv6].src
        else:
            continue

        try:
            qname_raw: bytes = dns[DNSQR].qname
            qtype = int(dns[DNSQR].qtype)
        except Exception:
            continue

        # Filtro 1: bytes crudos > 127 → corrupto (Base128 / binario)
        if _qname_has_nonascii(qname_raw):
            filtered += 1
            continue

        try:
            fqdn = qname_raw.decode("ascii", errors="replace").rstrip(".")
        except Exception:
            continue

        # Filtro 2: caracteres fuera de ASCII imprimible tras decode
        if not _fqdn_is_clean(fqdn):
            filtered += 1
            continue

        if not fqdn:
            continue

        records.append({
            "src_ip":    src_ip,
            "fqdn":      fqdn,
            "timestamp": float(pkt.time),
            "qtype":     qtype,
        })

    if filtered > 0:
        print(f"    {filtered:,} queries corruptas descartadas (bytes > 127)")
    return records

# Features agregadas (ventana deslizante)
def aggregated_features(window: List[dict]) -> dict:
    """Calcula las 8 features de ventana deslizante.

    Definiciones según Mendeley:
   
    time_avg    : media de los tiempos entre peticiones consecutivas (segundos)
    time_stdev  : desv. típica de los inter-arrival times (ddof=0)
    size_avg    : media de la longitud del FQDN en la ventana
    size_stdev  : desv. típica de las longitudes (ddof=0)
    throughput  : suma de longitudes / tiempo total de la ventana (bytes/s)
                  (usa len del FQDN como proxy del tamaño del paquete DNS)
    unique      : número de FQDNs únicos en la ventana
    entropy_avg : media de la entropía de Shannon en la ventana
    entropy_stdev: desv. típica de la entropía (ddof=0)
    """
    n      = len(window)
    sizes  = [r["len"]       for r in window]
    entrs  = [r["entropy"]   for r in window]
    times  = [r["timestamp"] for r in window]
    fqdns  = [r["fqdn"]      for r in window]

    size_avg      = float(np.mean(sizes))
    size_stdev    = float(np.std(sizes,  ddof=0))
    entropy_avg   = float(np.mean(entrs))
    entropy_stdev = float(np.std(entrs,  ddof=0))
    unique        = len(set(fqdns))

    if n >= 2:
        deltas     = [times[i] - times[i - 1] for i in range(1, n)]
        time_avg   = float(np.mean(deltas))
        time_stdev = float(np.std(deltas, ddof=0))
    else:
        time_avg   = 0.0
        time_stdev = 0.0

    time_span  = times[-1] - times[0] if n >= 2 else 0.0
    throughput = float(sum(sizes)) / time_span if time_span > 1e-9 else 0.0

    return {
        "time_avg":      time_avg,
        "time_stdev":    time_stdev,
        "size_avg":      size_avg,
        "size_stdev":    size_stdev,
        "throughput":    throughput,
        "unique":        unique,
        "entropy_avg":   entropy_avg,
        "entropy_stdev": entropy_stdev,
    }

# Procesar PCAP
def process_pcap(
    pcap_path:       str,
    label:           int,
    word_set:        Optional[set],
    window_size:     int,
    dns_type:        Optional[str] = None,
) -> pd.DataFrame:
    """Procesa un pcap y devuelve un DataFrame con las 22 columnas.

    - Descarta solo queries con bytes > 127 en el QNAME (corruptas).
    - Los dominios largos en Base32/Base64u se incluyen.
    - La ventana deslizante agrupa por (IP cliente, dominio registrado, QTYPE).
    """
    print(f"\nProcesando: {Path(pcap_path).name}")

    raw = extract_dns_queries(pcap_path)
    if not raw:
        print(f"    Sin queries DNS válidas.", file=sys.stderr)
        return pd.DataFrame(columns=CSV_COLUMN_NAMES)

    mixed_mode = (dns_type is None) or (dns_type.lower() == "mixed")

    if not mixed_mode:
        target = TYPE_TO_QTYPE.get(dns_type.upper())
        if target is None:
            print(f"    Tipo DNS '{dns_type}' desconocido. Procesando todo.",
                  file=sys.stderr)
            mixed_mode = True
        else:
            raw = [r for r in raw if r["qtype"] == target]
            if not raw:
                print(f"    Sin queries de tipo {dns_type}.", file=sys.stderr)
                return pd.DataFrame(columns=CSV_COLUMN_NAMES)

    # Estadísticas de qtypes en el pcap
    type_counts: Dict[str, int] = defaultdict(int)
    for r in raw:
        type_counts[QTYPE_NAMES.get(r["qtype"], str(r["qtype"]))] += 1
    print(f"    Queries aceptadas: {len(raw):,}")
    for t, c in sorted(type_counts.items()):
        print(f"      {t:<6}: {c:,}")

    # Calcular features individuales
    enriched: List[dict] = []
    for r in raw:
        fqdn       = r["fqdn"]
        feat       = individual_features(fqdn, word_set)
        qtype_name = QTYPE_NAMES.get(r["qtype"], str(r["qtype"]))
        reg_domain = get_registered_domain(fqdn)

        # group_key separa ventanas por tipo DNS en modo mixto
        if mixed_mode:
            group_key = f"{reg_domain}[{qtype_name}]"
        elif dns_type:
            group_key = f"{reg_domain}[{dns_type.upper()}]"
        else:
            group_key = reg_domain

        enriched.append({
            "src_ip":     r["src_ip"],
            "fqdn":       fqdn,
            "domain":     reg_domain,
            "_group_key": group_key,
            "timestamp":  r["timestamp"],
            **feat,
        })

    # Agrupar y ordenar por timestamp
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for rec in enriched:
        groups[(rec["src_ip"], rec["_group_key"])].append(rec)
    for key in groups:
        groups[key].sort(key=lambda x: x["timestamp"])

    # Ventana deslizante
    rows: List[dict] = []
    for (src_ip, _), recs in groups.items():
        n = len(recs)
        for i in range(n):
            window = recs[max(0, i - window_size + 1): i + 1]
            agg    = aggregated_features(window)
            rec    = recs[i]
            rows.append({
                "user_ip":          src_ip,
                "domain":           rec["domain"],
                "timestamp":        rec["timestamp"],
                "attack":           bool(label),
                "request":          rec["fqdn"],
                "len":              rec["len"],
                "subdomains_count": rec["subdomains_count"],
                "w_count":          rec["w_count"],
                "w_max":            rec["w_max"],
                "entropy":          rec["entropy"],
                "w_max_ratio":      rec["w_max_ratio"],
                "w_count_ratio":    rec["w_count_ratio"],
                "digits_ratio":     rec["digits_ratio"],
                "uppercase_ratio":  rec["uppercase_ratio"],
                **agg,
            })

    df = pd.DataFrame(rows, columns=CSV_COLUMN_NAMES)
    print(f"    Filas generadas: {len(df):,}")
    return df

# Deduplicación de filas
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas duplicadas exactas (excluye timestamp de la clave).

    El timestamp varía entre capturas del mismo tráfico, así que se excluye
    de la comparación. Se conserva la primera ocurrencia de cada duplicado.
    """
    dedup_cols = [c for c in CSV_COLUMN_NAMES if c != "timestamp"]
    antes = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    eliminadas = antes - len(df)
    if eliminadas > 0:
        print(f"    {eliminadas:,} duplicadas eliminadas ({antes:,} → {len(df):,})")
    else:
        print(f"    Sin duplicados ({antes:,} filas)")
    return df

# Busca los pcaps en la carpeta de capturas
def find_pcaps_in_capturas(capturas_dir: Path) -> List[str]:
    if not capturas_dir.exists():
        print(f"Directorio no encontrado: {capturas_dir}", file=sys.stderr)
        sys.exit(1)
    pcap_files: List[str] = []
    for ext in ("*.pcap", "*.pcapng"):
        pcap_files.extend(str(p) for p in capturas_dir.rglob(ext))
    pcap_files = sorted(set(pcap_files))
    if not pcap_files:
        print(f"No hay .pcap/.pcapng en {capturas_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(pcap_files)} pcap(s) en {capturas_dir}:")
    for p in pcap_files:
        size_mb = Path(p).stat().st_size / 1_048_576
        print(f"    {Path(p).name:<55}  ({size_mb:.2f} MB)")
    return pcap_files

def _print_stats(df: pd.DataFrame) -> None:
    total = len(df)
    if total == 0:
        print("    DataFrame vacío.")
        return
    attacks = df["attack"].astype(bool).sum()
    benign  = total - attacks
    print(f"\n  Distribución de clases")
    print(f"    Benigno (False): {benign:>10,}  ({100*benign/total:.1f}%)")
    print(f"    Ataque  (True):  {attacks:>10,}  ({100*attacks/total:.1f}%)")
    print(f"    Total:           {total:>10,}")

# Resuelve la ruta de salida del CSV
def resolve_output_path(filename: str) -> Path:
    p = Path(filename)
    if not p.is_absolute():
        p = OUTPUT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# Argumentos de línea de comandos
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extractor DNS compatible con Mendeley DOI 10.17632/c4n7fckkz3.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Todos los pcaps de capturas → un CSV (uso principal)\n"
            "  python pcap_a_dataset.py --from-capturas --label 1\n\n"
            "  # Carpeta capturas en otra ruta\n"
            "  python pcap_a_dataset.py --from-capturas \\\n"
            "      --capturas-dir /home/user/TFG/capturas --label 1\n\n"
            "  # PCap individual\n"
            "  python pcap_a_dataset.py --pcap iodine_TXT.pcap --label 1\n"
        ),
    )

    # Fuente de pcaps
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pcap", nargs="+", metavar="FICHERO",
        help="Uno o varios .pcap/.pcapng (acepta glob: capturas/*.pcap)",
    )
    src.add_argument(
        "--from-capturas", action="store_true", dest="from_capturas",
        help=f"Procesa TODOS los pcaps de --capturas-dir (default: {CAPTURAS_DIR})",
    )

    parser.add_argument(
        "--capturas-dir", default=str(CAPTURAS_DIR), dest="capturas_dir",
        metavar="DIR",
        help=f"Directorio de capturas (con --from-capturas). Default: {CAPTURAS_DIR}",
    )
    parser.add_argument("--label",     type=int, choices=[0, 1], required=True,
                        help="0 = benigno  /  1 = ataque")
    parser.add_argument("--out",       default="dataset_tfg.csv", metavar="CSV",
                        help=f"CSV de salida (relativo → {OUTPUT_DIR}/)")
    parser.add_argument("--dns-type",  default=None, dest="dns_type", metavar="TIPO",
                        help="TXT | NULL | A → filtra QTYPE. Omitir → todos.")
    parser.add_argument("--window",    type=int, default=DEFAULT_WINDOW_SIZE, metavar="N",
                        help=f"Tamaño ventana deslizante (default: {DEFAULT_WINDOW_SIZE})")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolver lista de pcaps
    if args.from_capturas:
        pcap_files = find_pcaps_in_capturas(Path(args.capturas_dir))
    else:
        pcap_files = []
        for pattern in args.pcap:
            expanded = glob.glob(pattern)
            pcap_files.extend(expanded if expanded else [pattern])
        pcap_files = sorted(set(pcap_files))

    if not pcap_files:
        print("No se encontraron ficheros pcap.", file=sys.stderr)
        sys.exit(1)

    out_path    = resolve_output_path(args.out)
    dns_type    = args.dns_type
    mixed_mode  = (dns_type is None) or (dns_type.lower() == "mixed")

    print("pcap_a_dataset.py — compatible con Mendeley DOI 10.17632/c4n7fckkz3.3")
    print(f"  Fuente: {'carpeta ' + args.capturas_dir if args.from_capturas else 'pcaps individuales'} ({len(pcap_files)} pcap)")
    print(f"  Label: {'1 (ataque)' if args.label else '0 (benigno)'}  Modo DNS: {'mixto' if mixed_mode else dns_type.upper()}  Ventana: {args.window}")
    print(f"  Salida: {out_path}")

    # Cargar léxico desde WORDS_PATH
    word_set = load_english_words()

    # Procesar cada pcap 
    dfs: List[pd.DataFrame] = []
    for pcap_path in pcap_files:
        df = process_pcap(
            pcap_path       = pcap_path,
            label           = args.label,
            word_set        = word_set,
            window_size     = args.window,
            dns_type        = dns_type,
        )
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("\nNo se generaron datos.", file=sys.stderr)
        sys.exit(1)

    result = pd.concat(dfs, ignore_index=True)

    # Deduplicación
    print(f"\nTotal filas (todos los pcaps): {len(result):,}")
    result = deduplicate(result)

    result.to_csv(out_path, index=False)
    print(f"\nCSV guardado: {out_path}  ({len(result):,} filas)")
    _print_stats(result)

    print("\nColumnas del CSV:")
    labels = {**{i: "meta" for i in range(5)},
              **{i: "individual" for i in range(5, 14)},
              **{i: "agregada"   for i in range(14, 22)}}
    for i, col in enumerate(CSV_COLUMN_NAMES):
        print(f"    [{i:02d}] {col:<20}  ({labels[i]})")

    print(f"\nCompletado. Ficheros en: {out_path.parent}")


if __name__ == "__main__":
    main()
