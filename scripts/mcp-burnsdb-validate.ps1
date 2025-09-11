param(
  [string]$Url = "https://BurnsDB.fastmcp.app/mcp"
)

Write-Host "Validating MCP server at $Url" -ForegroundColor Cyan

# Pin versions for reproducibility
$npx = "npx -y"
$remote = "mcp-remote@0.1.29"
$inspector = "@modelcontextprotocol/inspector@0.16.6"

# Ping via inspector (tools/list)
Write-Host "Listing tools via inspector..." -ForegroundColor Yellow
cmd /c "npx -y $inspector --cli --transport stdio --method tools/list -- npx -y $remote $Url" 2>&1 | Out-String | Write-Output

# Quick tool call example (echo)
Write-Host "Calling echo via mcp-remote..." -ForegroundColor Yellow
cmd /c "npx -y $remote $Url" 2>&1 | Out-String | Write-Output
