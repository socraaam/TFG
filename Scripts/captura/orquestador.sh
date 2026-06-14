#!/bin/bash
# orquestador.sh — Orquestador de túneles DNS con iodine

# Ejecuta automáticamente 3 sesiones: TXT → NULL → A
# Coordina servidor (local) y cliente (remoto).

# Uso:
#   1. En el servidor: ./orquestador.sh
#   2. En el cliente (cuando se indique):
#      ./cliente_dns.sh TXT   (luego NULL, luego A)

TIPOS_DNS=("TXT" "NULL" "A")
SERVIDOR_SCRIPT="./servidor_dns.sh"
BASE_DIR="/home/quasar/Documents/TFG"
PCAP_DIR="$BASE_DIR/capturas"
LOG_DIR="$BASE_DIR/logs"
TIMEOUT_SESION=3600

mkdir -p "$PCAP_DIR" "$LOG_DIR"

echo "[ORQ] Iniciando captura: ${TIPOS_DNS[*]}"

for TIPO in "${TIPOS_DNS[@]}"; do
    echo "[ORQ] Sesión: $TIPO"
    READY_FLAG="/tmp/servidor_listo_${TIPO}"
    rm -f "$READY_FLAG"
    bash "$SERVIDOR_SCRIPT" "$TIPO" &
    SERVIDOR_PID=$!
    intentos=0
    while [[ ! -f "$READY_FLAG" ]]; do
        sleep 1
        intentos=$((intentos + 1))
        if (( intentos >= 20 )); then
            echo "[ORQ] Timeout servidor $TIPO"
            kill $SERVIDOR_PID 2>/dev/null
            wait $SERVIDOR_PID 2>/dev/null
            break
        fi
        if ! kill -0 $SERVIDOR_PID 2>/dev/null; then
            echo "[ORQ] Servidor $TIPO terminó inesperadamente"
            break
        fi
    done
    if [[ ! -f "$READY_FLAG" ]]; then
        continue
    fi
    echo "[ORQ] Ejecuta en cliente: ./cliente_dns.sh $TIPO"
    tiempo=0
    while kill -0 $SERVIDOR_PID 2>/dev/null; do
        sleep 5
        tiempo=$((tiempo + 5))
        if (( tiempo >= TIMEOUT_SESION )); then
            kill $SERVIDOR_PID 2>/dev/null
            break
        fi
    done
    wait $SERVIDOR_PID 2>/dev/null
    PCAP=$(ls "$PCAP_DIR/iodine_dns_${TIPO}_"*.pcap 2>/dev/null | tail -1)
    if [[ -f "$PCAP" ]]; then
        echo "[ORQ] ✓ $TIPO: $(basename "$PCAP")"
    fi
    if [[ "$TIPO" != "${TIPOS_DNS[-1]}" ]]; then
        sleep 1
    fi
done

echo "[ORQ] Completo. Próximo paso: python pcap_a_dataset.py --from-capturas --label 1"
