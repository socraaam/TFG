#!/bin/bash
# Túnel DNS con iodine (servidor)
# Levanta servidor iodine y captura con tcpdump

TUNEL_IP="10.0.0.1"
DOMINIO="tfg.local"
PASSWORD="tfg2026"
BASE_DIR="/home/quasar/Documents/TFG"
RECEIVED_DIR="$BASE_DIR/ficheros_recibidos"
PCAP_DIR="$BASE_DIR/capturas"

DNS_TYPE="${1^^}"
if [[ -z "$DNS_TYPE" ]]; then
    echo "Error de uso"
    exit 1
fi
if [[ ! "$DNS_TYPE" =~ ^(NULL|TXT|A)$ ]]; then
    echo "Tipo no soportado: '$DNS_TYPE'"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCAP="$PCAP_DIR/iodine_dns_${DNS_TYPE}_${TIMESTAMP}.pcap"
RECEIVED_DIR_TIPO="$RECEIVED_DIR/$DNS_TYPE"
mkdir -p "$RECEIVED_DIR_TIPO" "$PCAP_DIR"

echo "Iniciando:$DNS_TYPE → $PCAP"

# Limpieza
pkill -KILL iodined 2>/dev/null
sleep 0.5
ip link set dns0 down 2>/dev/null
ip link delete dns0 2>/dev/null
ip tuntap del dev dns0 mode tun 2>/dev/null
sleep 0.5
if ip link show dns0 &>/dev/null; then
    echo "No se pudo limpiar dns0"
    exit 1
fi

# Levantar iodine
iodined -f -c -P "$PASSWORD" "$TUNEL_IP" "$DOMINIO" &
IODINE_PID=$!
intentos=0
until ip link show dns0 &>/dev/null && kill -0 $IODINE_PID 2>/dev/null; do
    sleep 1
    intentos=$((intentos + 1))
    if (( intentos >= 15 )); then
        echo "iodined no levantó"
        kill $IODINE_PID 2>/dev/null
        exit 1
    fi
done

# Captura
tcpdump -i eth0 port 53 and host 192.168.1.142 -w "$PCAP" 2>/dev/null &
TCPDUMP_PID=$!
intentos=0
until [[ -f "$PCAP" ]]; do
    sleep 0.5
    intentos=$((intentos + 1))
    if (( intentos >= 10 )); then
        break
    fi
done

READY_FLAG="/tmp/servidor_listo_${DNS_TYPE}"
touch "$READY_FLAG"

RUNNING=true

cleanup() {
    RUNNING=false
    kill $IODINE_PID 2>/dev/null
    wait $IODINE_PID 2>/dev/null
    ip link set dns0 down 2>/dev/null
    ip link delete dns0 2>/dev/null
    ip tuntap del dev dns0 mode tun 2>/dev/null
    kill -INT $TCPDUMP_PID 2>/dev/null
    wait $TCPDUMP_PID 2>/dev/null
    rm -f "$READY_FLAG"
    exit 0
}
trap cleanup INT TERM

reconstruir() {
    local file="$1"
    ls "$RECEIVED_DIR_TIPO/${file}_chunk_"*.b64 2>/dev/null \
        | sort -V \
        | while read -r c; do cat "$c"; done \
        > "$RECEIVED_DIR_TIPO/$file.full.b64"
    if [[ ! -s "$RECEIVED_DIR_TIPO/$file.full.b64" ]]; then
        rm -f "$RECEIVED_DIR_TIPO/$file.full.b64"
        return 1
    fi
    if base64 -d "$RECEIVED_DIR_TIPO/$file.full.b64" > "$RECEIVED_DIR_TIPO/$file" 2>/dev/null; then
        echo "Reconstruido exitosamente $file"
    fi
    rm -f "$RECEIVED_DIR_TIPO/${file}_chunk_"*.b64 "$RECEIVED_DIR_TIPO/$file.full.b64"
}

while $RUNNING; do
    tmpfile=$(mktemp)
    nc -l -s "$TUNEL_IP" -p 1234 -q 1 > "$tmpfile"
    $RUNNING || { rm -f "$tmpfile"; break; }
    header=$(head -n 1 "$tmpfile")
    if [[ "$header" =~ \[\[CHUNK\ (.*)\ ([0-9]+)\ ([0-9]+)\]\] ]]; then
        filename="${BASH_REMATCH[1]}"
        id="${BASH_REMATCH[2]}"
        total="${BASH_REMATCH[3]}"
        if [[ "$filename" == "__FIN_SESION__" ]]; then
            rm -f "$tmpfile"
            cleanup
        fi
        outfile="$RECEIVED_DIR_TIPO/${filename}_chunk_${id}.b64"
        sed '1d;$d' "$tmpfile" > "$outfile"
        if [[ "$id" == "$total" ]]; then
            reconstruir "$filename"
        fi
    fi
    rm -f "$tmpfile"
done
