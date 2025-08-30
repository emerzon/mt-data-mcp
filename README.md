# MetaTrader5 MCP Server

A Model Context Protocol (MCP) server that provides access to MetaTrader5 market data and trading functionality.

## Features

- Connect to MetaTrader5 terminal
- Retrieve symbol information and market data
- Get historical rates (OHLCV) for any timeframe
- Access tick data and market depth (DOM)
- Real-time market data streaming
- Support for all major trading instruments

## Installation

1. Install MetaTrader5 terminal
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure your MetaTrader5 credentials:

```bash
cp .env.example .env
```

Edit the `.env` file with your broker credentials (optional - you can also connect without pre-configured credentials).

## Usage

### Starting the MCP Server

```bash
python server.py
```

### Available Tools

#### Market Data
- `get_symbols(search_term, limit)` - Smart search for trading symbols (CSV output)
- `get_symbol_groups(search_term, limit)` - Get available symbol groups (CSV output)
- `get_symbol_info(symbol)` - Get detailed symbol information
- `get_rates(symbol, timeframe, candles, start_datetime, end_datetime)` - Get historical OHLCV data in CSV format
- `get_ticks(symbol, count, start_datetime)` - Get tick data in CSV format
- `get_market_depth(symbol)` - Get market depth (DOM)

### Example Usage

The server automatically connects to MetaTrader5 when any tool is called. Connection credentials can be configured via environment variables or the server will use the terminal's current login.

```python
# Get all available symbol groups from MT5
groups = get_symbol_groups()

# Get all visible symbols (no search term)
all_symbols = get_symbols()

# Smart search: Find symbols containing "EUR" (EURUSD, JPYEUR, etc.)
eur_symbols = get_symbols(search_term="EUR", limit=20)

# Search by group: Find all major forex pairs
majors = get_symbols(search_term="Majors", limit=10)

# Limit number of results
eur_top5 = get_symbols(search_term="EUR", limit=5)

# Get symbol information
info = get_symbol_info("GBPUSD")

# Get EUR/USD hourly data
rates = get_rates("EURUSD", "H1", 100)

# Get tick data
ticks = get_ticks("EURUSD", 50)
```

## Intelligent Symbol Search

The `get_symbols` function uses a smart 3-tier search strategy:

Search order:
1. Group name match (e.g., `Majors`, `Forex`)
2. Symbol name match (e.g., `EUR` → EURUSD, JPYEUR, ...)
3. Description match (symbol or group path)

Parameters:
- `search_term`: Optional search term. If omitted, lists visible symbols.
- `limit`: Optional maximum number of symbols to return.

Response format:
- CSV with header `name,group,description` and one row per symbol.

## Date Inputs

Flexible date/time parsing powered by `dateparser`:
- Accepts natural language (e.g., `yesterday 14:00`, `2 days ago`)
- Accepts common formats (e.g., `2025-08-29`, `2025/08/29 14:30`, `2025-08-29 14:30 UTC`)
- Times are interpreted and converted to UTC internally

Usage patterns for `get_rates`:
- Only `start_datetime`: returns bars forward from the start (up to `candles`).
- Only `end_datetime`: returns the last `candles` bars before the end.
- Both `start_datetime` and `end_datetime`: returns bars within the range (ignores `candles`).

## Quote Data Format

Historical rates and tick data are returned in compact CSV format with intelligent column filtering:

### Historical Rates (`get_rates`)
**Core columns**: `time,open,high,low,close` (always included)  
**Optional columns**: `tick_volume,spread,real_volume` (included only if they contain meaningful data)

```csv
time,open,high,low,close,tick_volume
2025-08-29T14:00:00,1.16945,1.17072,1.16937,1.17017,2690
2025-08-29T15:00:00,1.17017,1.17089,1.16964,1.16986,2156
```
*Note: `spread` and `real_volume` columns excluded because they contained only zeros*

### Tick Data (`get_ticks`)
**Core columns**: `time,bid,ask` (always included)  
**Optional columns**: `last,volume,flags` (included only if they contain meaningful data)

```csv
time,bid,ask
2025-08-29T15:30:15.123,1.16983,1.16985
2025-08-29T15:30:16.456,1.16982,1.16986
```
*Note: `last`, `volume`, and `flags` columns excluded because they contained no meaningful data*

### Intelligent Column Filtering
- **Meaningful Data Check**: Columns are included only if they have at least one non-zero value OR multiple different values
- **Space Efficiency**: Empty/constant columns are automatically excluded to reduce response size
- **Consistency**: Core OHLC columns (rates) and bid/ask columns (ticks) are always included

### Response Format
Both functions return:
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "H1",  // rates only
  "candles": 100,
  "csv_data": "time,open,high,low,close,tick_volume\n2025-08-29T14:00:00,1.16945,1.17072,1.16937,1.17017,2690\n..."
}
```

## Timeframes

Supported timeframes for historical data (MetaTrader5):
- Minutes: `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`, `M12`, `M15`, `M20`, `M30`
- Hours: `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`
- Days/Weeks/Months: `D1`, `W1`, `MN1`

## Requirements

- Windows OS (MetaTrader5 requirement)
- MetaTrader5 terminal installed
- Python 3.8+
- Active internet connection for real-time data

## Error Handling

All functions return structured responses with success/error status:

```python
{
    "success": True/False,
    "data": {...},  # on success
    "error": "Error message"  # on failure
}
```

## Security Notes

- Store credentials securely in environment variables
- Use demo accounts for testing
- Never commit credentials to version control
- Consider using read-only API access where possible

## License

MIT License
