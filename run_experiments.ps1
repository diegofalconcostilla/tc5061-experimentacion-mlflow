# Ejecuta las corridas del experimento TC5061-diamonds-precio-experimentacion.
# Diseño: se parte de la configuración por defecto y se cambia un hiperparámetro a la vez,
# para poder atribuir cada diferencia en las métricas a ese hiperparámetro.
# Uso (desde la carpeta de la actividad): .\run_experiments.ps1

$python = ".\.venv\Scripts\python.exe"

$corridas = @(
    @("--model", "linear"),                                     # referencia: baseline de la Actividad 2
    @(),                                                        # configuración por defecto
    @("--run-name", "rf-d16-l2-mf0.5-n200-rep"),               # misma configuración: prueba de reproducibilidad
    @("--max-depth", "4"),
    @("--max-depth", "8"),
    @("--max-depth", "0", "--min-samples-leaf", "1", "--max-features", "1.0"),
    @("--n-estimators", "50"),
    @("--min-samples-leaf", "20"),
    @("--max-features", "1.0"),
    @("--log-target")
)

foreach ($argumentos in $corridas) {
    Write-Host "python src/train.py $($argumentos -join ' ')"
    & $python src/train.py @argumentos
    if ($LASTEXITCODE -ne 0) { throw "La corrida falló: $($argumentos -join ' ')" }
}
