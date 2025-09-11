Param(
  [string]$ServerBase = $env:BDB_SERVER_BASE
)

if (-not $ServerBase -or $ServerBase.Trim() -eq "") {
  $ServerBase = "https://burnsdb.fastmcp.app"
}

Write-Host "[BDB] Querying hosted tool definition from: $ServerBase/openai/hosted-tool" -ForegroundColor Cyan

try {
  $tool = Invoke-RestMethod -Uri ("{0}/openai/hosted-tool" -f $ServerBase.TrimEnd('/')) -Method GET -TimeoutSec 15
} catch {
  Write-Error "Failed to fetch hosted tool: $($_.Exception.Message)"
  exit 1
}

Write-Host "[BDB] Hosted tool JSON:" -ForegroundColor Green
$tool | ConvertTo-Json -Depth 5

if (-not $tool.url) {
  Write-Warning "Tool JSON missing 'url' field; cannot probe connectivity."
  exit 0
}

$probe = ("{0}/" -f $tool.url.TrimEnd('/'))
Write-Host "[BDB] Probing MCP root: $probe" -ForegroundColor Cyan

try {
  $ok = Invoke-RestMethod -Uri $probe -Method GET -TimeoutSec 10
  Write-Host "[BDB] MCP root response:" -ForegroundColor Green
  $ok | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "Probe failed: $($_.Exception.Message)"
}

Write-Host "[BDB] Health check: $($tool.url.TrimEnd('/'))/health" -ForegroundColor Cyan
try {
  $health = Invoke-RestMethod -Uri ("{0}/health" -f $tool.url.TrimEnd('/')) -Method GET -TimeoutSec 10
  Write-Host "[BDB] MCP health:" -ForegroundColor Green
  $health | ConvertTo-Json -Depth 5
} catch {
  Write-Warning "Health check failed: $($_.Exception.Message)"
}

Write-Host "Done." -ForegroundColor Cyan

