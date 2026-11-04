#!/bin/sh
# Se ejecuta dentro de un pod cliente. Uso: sh -s N  (N peticiones al Service)
N="${1:-30}"
BODY='{"ingreso_mensual":1800,"deuda_actual":3500,"antiguedad_laboral":4,"monto_solicitado":9000,"plazo_meses":24}'
i=0
while [ "$i" -lt "$N" ]; do
  R=$(curl -s -m 3 -X POST http://credito-service/predecir -H 'Content-Type: application/json' -d "$BODY")
  P=$(echo "$R" | grep -o '"pod":"[^"]*"' | cut -d'"' -f4)
  echo "${P:-FALLO}"
  i=$((i+1))
done
