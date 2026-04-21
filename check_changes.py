import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", 
     "import sys; sys.path.insert(0, 'src'); from mtdata.core.trading.use_cases import run_trade_place; import inspect; lines = inspect.getsource(run_trade_place).split('\\n'); print([l for l in lines[40:60] if 'volume' in l.lower()])"],
    cwd="C:\\Users\\Admin\\Documents\\Code\\mtdata",
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)
