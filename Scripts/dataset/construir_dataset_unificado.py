"""
construir_dataset_unificado.py — Genera dataset_unificado.csv
  - Lowercasing universal antes de cualquier feature
  - 13 features
  - Muestreo estratificado de dataset.csv (scan 70 chunks, samplea del pool completo)
  - Topes proporcionales por familia (~86,596 ataques por familia)

Rutas:
  DATA_ROOT = datos crudos externos (no versionados), configurable con la
              variable de entorno TFG_DATA_ROOT.
  OUT_DIR   = <raiz_del_proyecto>/Datasets

Uso:
    python -m Scripts.dataset.construir_dataset_unificado
"""

import os, re, sys, logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.constantes import FEATURES_13 as FEATURES

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TFG_DATA_ROOT", r"C:/Users/Marcos/Desktop/Compartida_ubuntu/TFG"))
OUT_DIR   = ROOT / "Datasets"
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CAPS = {
    # word_based
    "mendeley":             40_000,
    "kaggle_dnsexfil_mod":  46_596,
    # base32
    "propio":               16_021,
    "kaggle_iodine":        29_407,
    "kaggle_dnsexfil":      29_407,
    "mendeley_encoded":     11_763,
    # encoded
    "kaggle_cobaltstrike":  15_658,
    "kaggle_dns2tcp":       15_658,
    "kaggle_dnscat2":       15_658,
    "kaggle_dnspot":         8_309,
    "kaggle_ozymandns":     15_658,
    "kaggle_tcp_over_dns":  15_658,
    # benign
    "mendeley_benign":     120_000,
    "kaggle_benign":       100_000,
}

MENDELEY_COLS = [
    "user_ip", "domain", "timestamp", "attack", "request",
    "len", "subdomains_count", "w_count", "w_max", "entropy",
    "w_max_ratio", "w_count_ratio", "digits_ratio", "uppercase_ratio",
    "time_avg", "time_stdev", "size_avg", "size_stdev",
    "throughput", "unique", "entropy_avg", "entropy_stdev",
]

KAGGLE_TOOLS = {
    "kaggle_dnsexfil_mod":  "malicious/dnsexfiltrator_modified/dnsexfiltrator_modified.csv",
    "kaggle_dnsexfil":      "malicious/dnsexfiltrator/dnsexfiltrator.csv",
    "kaggle_iodine":        "malicious/iodine/iodine.csv",
    "kaggle_cobaltstrike":  "malicious/cobaltstrike/cobaltstrike.csv",
    "kaggle_dns2tcp":       "malicious/dns2tcp/dns2tcp.csv",
    "kaggle_dnscat2":       "malicious/dnscat2/dnscat2.csv",
    "kaggle_dnspot":        "malicious/dnspot/dnspot.csv",
    "kaggle_ozymandns":     "malicious/ozymandns/ozymandns.csv",
    "kaggle_tcp_over_dns":  "malicious/tcp_over_dns/tcp_over_dns.csv",
    "kaggle_benign":        "benign/benign.csv",
}

VOWELS = set("aeiou")


def _shannon(s: str) -> float:
    if not s: return 0.0
    cnt = Counter(s)
    t = len(s)
    return -sum((v/t) * np.log2(v/t) for v in cnt.values())


def compute_features(requests: pd.Series, domains: pd.Series) -> pd.DataFrame:
    """Calcula las 13 features sobre el FQDN normalizado a minusculas."""
    req = requests.astype(str).str.strip().str.rstrip(".").str.lower()
    dom = domains.astype(str).str.strip().str.rstrip(".").str.lower()

    fqdn_len      = req.str.len()
    dom_len       = dom.str.len()
    subdomain_len = (fqdn_len - dom_len - 1).clip(lower=0)
    safe_len      = fqdn_len.replace(0, np.nan)

    labels_list   = req.str.split(".")
    n_labels      = labels_list.apply(len)
    max_label     = labels_list.apply(lambda p: max((len(x) for x in p), default=0))
    entropy_s     = req.apply(_shannon)

    digits_count  = req.apply(lambda s: sum(c.isdigit() for c in s))
    vowel_count   = req.apply(lambda s: sum(c in VOWELS for c in s))
    special_count = req.apply(lambda s: s.count(".") + s.count("-"))
    run_segs      = req.apply(lambda s: max(
                       (len(seg) for seg in re.split(r"[.\-]", s)), default=0))

    safe_labels = n_labels.replace(0, np.nan)

    return pd.DataFrame({
        "fqdn_len":            fqdn_len,
        "subdomain_len":       subdomain_len,
        "labels":              n_labels,
        "max_label":           max_label,
        "entropy":             entropy_s,
        "digits_ratio":        (digits_count / safe_len).fillna(0.0),
        "subdomain_fraction":  (subdomain_len / safe_len).fillna(0.0),
        "entropy_x_subdomain": entropy_s * subdomain_len,
        "max_label_fraction":  (max_label / safe_len).fillna(0.0),
        "avg_label_len":       ((fqdn_len - n_labels + 1) / safe_labels).fillna(0.0),
        "special_ratio":       (special_count / safe_len).fillna(0.0),
        "alpha_run_ratio":     (run_segs / safe_len).fillna(0.0),
        "vowel_ratio":         (vowel_count / safe_len).fillna(0.0),
    })


def load_mendeley_modified(path: Path) -> pd.DataFrame:
    """Ataques word_based de dataset_modified.csv."""
    df = pd.read_csv(path, header=None, names=MENDELEY_COLS, low_memory=False)
    df["attack_b"] = df["attack"].astype(str).isin(["1", "True", "1.0"]).astype(int)
    atk = df[df["attack_b"] == 1].copy()
    cap = CAPS["mendeley"]
    if len(atk) > cap:
        atk = atk.sample(cap, random_state=SEED)
    feats = compute_features(atk["request"], atk["domain"])
    feats["attack"] = 1
    feats["source"] = "mendeley"
    return feats


def load_mendeley_stratified(path: Path) -> pd.DataFrame:
    """Muestreo estratificado de dataset.csv recorriendo todos los chunks."""
    atk_records, ben_records = [], []
    for chunk in pd.read_csv(path, header=None, names=MENDELEY_COLS,
                              chunksize=500_000, low_memory=False):
        atk_mask = chunk["attack"].astype(str).isin(["1", "True", "1.0"])
        atk_records.append(chunk[atk_mask][["request", "domain"]].copy())
        ben_records.append(chunk[~atk_mask][["request", "domain"]].copy())

    atk_all = pd.concat(atk_records, ignore_index=True).dropna(subset=["request", "domain"])
    ben_all = pd.concat(ben_records, ignore_index=True).dropna(subset=["request", "domain"])

    rng = np.random.default_rng(SEED)

    # Muestra de ataque (mendeley_encoded)
    cap_atk = CAPS["mendeley_encoded"]
    if len(atk_all) > cap_atk:
        atk_sample = atk_all.iloc[rng.choice(len(atk_all), size=cap_atk, replace=False)].copy()
    else:
        atk_sample = atk_all.copy()
    feats_atk = compute_features(atk_sample["request"], atk_sample["domain"])
    feats_atk["attack"] = 1
    feats_atk["source"] = "mendeley_encoded"

    # Muestra benigna
    cap_ben = CAPS["mendeley_benign"]
    if len(ben_all) > cap_ben:
        ben_sample = ben_all.iloc[rng.choice(len(ben_all), size=cap_ben, replace=False)].copy()
    else:
        ben_sample = ben_all.copy()
    feats_ben = compute_features(ben_sample["request"], ben_sample["domain"])
    feats_ben["attack"] = 0
    feats_ben["source"] = "mendeley_benign"

    return pd.concat([feats_atk, feats_ben], ignore_index=True)


def load_propio(atk_path: Path, ben_path: Path) -> pd.DataFrame:
    """Ataques y benignos del dataset propio (iodine)."""
    atk = pd.read_csv(atk_path, low_memory=False)
    ben = pd.read_csv(ben_path, low_memory=False)
    req_col = next(c for c in atk.columns if c.lower() == "request")
    dom_col = "domain"

    cap = CAPS["propio"]
    if len(atk) > cap:
        atk = atk.sample(cap, random_state=SEED)
    feats_atk = compute_features(atk[req_col], atk[dom_col])
    feats_atk["attack"] = 1
    feats_atk["source"] = "propio"

    feats_ben = compute_features(ben[req_col], ben[dom_col])
    feats_ben["attack"] = 0
    feats_ben["source"] = "propio"

    return pd.concat([feats_atk, feats_ben], ignore_index=True)


def load_kaggle_tool(kaggle_dir: Path, label: str, rel_path: str, is_benign: bool) -> pd.DataFrame:
    """Carga una herramienta de Kaggle y calcula sus features."""
    df = pd.read_csv(kaggle_dir / rel_path,
                     usecols=["dns_domain_name", "dns_second_level_domain"],
                     low_memory=False)
    df = df.dropna(subset=["dns_domain_name"])
    cap = CAPS.get(label)
    if cap is not None and len(df) > cap:
        df = df.sample(cap, random_state=SEED)
    feats = compute_features(df["dns_domain_name"], df["dns_second_level_domain"])
    feats["attack"] = 0 if is_benign else 1
    feats["source"] = label
    return feats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DIR = DATA_ROOT / "dataset_kaggle"
    MEN_DIR    = DATA_ROOT / "dataset_mendeley"
    OWN_DIR    = DATA_ROOT / "dataset_out"

    parts = []

    log.info("Cargando word_based...")
    parts.append(load_mendeley_modified(MEN_DIR / "dataset_modified.csv"))
    parts.append(load_kaggle_tool(
        KAGGLE_DIR, "kaggle_dnsexfil_mod",
        KAGGLE_TOOLS["kaggle_dnsexfil_mod"], is_benign=False))

    log.info("Cargando base32 y benignos de mendeley...")
    parts.append(load_mendeley_stratified(MEN_DIR / "dataset.csv"))
    parts.append(load_propio(
        OWN_DIR / "dataset_ataque.csv",
        OWN_DIR / "dataset_benigno.csv"))
    for label in ["kaggle_iodine", "kaggle_dnsexfil"]:
        parts.append(load_kaggle_tool(
            KAGGLE_DIR, label, KAGGLE_TOOLS[label], is_benign=False))

    log.info("Cargando encoded...")
    for label in ["kaggle_cobaltstrike", "kaggle_dns2tcp", "kaggle_dnscat2",
                  "kaggle_dnspot", "kaggle_ozymandns", "kaggle_tcp_over_dns"]:
        parts.append(load_kaggle_tool(
            KAGGLE_DIR, label, KAGGLE_TOOLS[label], is_benign=False))

    parts.append(load_kaggle_tool(
        KAGGLE_DIR, "kaggle_benign", KAGGLE_TOOLS["kaggle_benign"], is_benign=True))

    cols_out = ["attack", "source"] + FEATURES
    df_common = pd.concat(parts, ignore_index=True)
    df_common = df_common[[c for c in cols_out if c in df_common.columns]]

    out_principal = OUT_DIR / "dataset_unificado.csv"
    df_common.to_csv(out_principal, index=False)
    log.info(f"Guardado: {out_principal}  ({len(df_common):,} filas)")


if __name__ == "__main__":
    main()
