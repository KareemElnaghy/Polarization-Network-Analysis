# Restart the Ollama server with a given parallel-slot count (Windows). usage: powershell -File scripts/llm/start_ollama.ps1 -Parallel 8
param([int]$Parallel = 4)
Get-Process "ollama app", ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 3
$env:OLLAMA_NUM_PARALLEL = "$Parallel"
$env:OLLAMA_MAX_LOADED_MODELS = "1"       # never two models at once
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_CONTEXT_LENGTH = "4096"       # per slot; explicit so it does not depend on VRAM-based defaults
$env:OLLAMA_KEEP_ALIVE = "30m"
New-Item -ItemType Directory -Force logs_llm | Out-Null
Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden -RedirectStandardOutput "logs_llm/ollama_serve.out" -RedirectStandardError "logs_llm/ollama_serve.err"
Start-Sleep 5
(Invoke-RestMethod http://localhost:11434/api/version).version
