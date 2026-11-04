# Genera las 4 evidencias exigidas de Kubernetes en docs/evidencias/.
# Las peticiones se lanzan DESDE DENTRO del clúster contra el Service (balanceo real
# de kube-proxy; `kubectl port-forward` fijaría un solo pod y no demostraría nada).
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$out = "docs/evidencias"
New-Item -ItemType Directory -Force $out | Out-Null

function Log($file, $cmd) {
    Add-Content -Encoding utf8 "$out/$file" "PS> $cmd"
    $res = Invoke-Expression $cmd 2>&1 | Out-String
    Add-Content -Encoding utf8 "$out/$file" $res
}
function Peticiones($n) {
    # n peticiones al Service desde un pod cliente efimero; devuelve el pod que atendio cada una
    kubectl delete pod cliente --ignore-not-found --wait=true | Out-Null
    $sh = (Get-Content scripts/cliente.sh -Raw) -replace "`r", ""
    $sh | kubectl run cliente --rm -i --restart=Never --image=curlimages/curl:8.10.1 --command -- sh -s $n 2>$null |
        Where-Object { $_ -match '^(credito-api|FALLO)' }
}

Remove-Item "$out/k8s_*.txt" -ErrorAction SilentlyContinue

# --- 1. >=3 réplicas Running ---------------------------------------------------
Log "k8s_1_replicas_running.txt" "kubectl get deployment credito-api"
Log "k8s_1_replicas_running.txt" "kubectl get pods -l app=credito-api -o wide"

# --- 2. Balanceo de carga ------------------------------------------------------
Add-Content -Encoding utf8 "$out/k8s_2_balanceo.txt" "PS> 30 peticiones POST /predecir al Service (desde un pod cliente)"
$r = Peticiones 30
Add-Content -Encoding utf8 "$out/k8s_2_balanceo.txt" ($r | Out-String)
Add-Content -Encoding utf8 "$out/k8s_2_balanceo.txt" "Resumen de pods que atendieron (conteo):"
Add-Content -Encoding utf8 "$out/k8s_2_balanceo.txt" (($r | Where-Object { $_ } | Group-Object | ForEach-Object { "{0,3}  {1}" -f $_.Count, $_.Name }) | Out-String)

# --- 3. Autorreparación --------------------------------------------------------
$f = "k8s_3_autorreparacion.txt"
Log $f "kubectl get pods -l app=credito-api"
$victima = (kubectl get pods -l app=credito-api -o jsonpath='{.items[0].metadata.name}')
Log $f "kubectl delete pod $victima --wait=false"
Add-Content -Encoding utf8 "$out/$f" "PS> 20 peticiones MIENTRAS el pod se recrea"
$r = Peticiones 20
Add-Content -Encoding utf8 "$out/$f" (($r | Group-Object | ForEach-Object { "{0,3}  {1}" -f $_.Count, $_.Name }) | Out-String)
Add-Content -Encoding utf8 "$out/$f" ("Peticiones vacias/fallidas: " + (@($r | Where-Object { -not $_ }).Count))
Start-Sleep 25
Log $f "kubectl get pods -l app=credito-api"

# --- 4. Escalado ---------------------------------------------------------------
$f = "k8s_4_escalado.txt"
Log $f "kubectl scale deployment credito-api --replicas=5"
kubectl rollout status deployment/credito-api --timeout=180s | Out-Null
Log $f "kubectl get pods -l app=credito-api"
$r = Peticiones 40
Add-Content -Encoding utf8 "$out/$f" "PS> 40 peticiones con 5 replicas - pods que respondieron:"
Add-Content -Encoding utf8 "$out/$f" (($r | Where-Object { $_ } | Group-Object | ForEach-Object { "{0,3}  {1}" -f $_.Count, $_.Name }) | Out-String)
Log $f "kubectl scale deployment credito-api --replicas=3"
kubectl rollout status deployment/credito-api --timeout=180s | Out-Null
Start-Sleep 10
Log $f "kubectl get pods -l app=credito-api"
