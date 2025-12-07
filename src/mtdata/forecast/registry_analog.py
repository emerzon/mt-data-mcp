
def _register_analog_methods(add_func):
    """Register Analog forecasting methods."""
    add_func("analog", "Nearest-neighbor search based on historical patterns", [
        {"name": "window_size", "type": "int", "description": "Length of pattern to match (default: 64)"},
        {"name": "search_depth", "type": "int", "description": "Bars back to search (default: 5000)"},
        {"name": "top_k", "type": "int", "description": "Number of analogs (default: 20)"},
        {"name": "metric", "type": "str", "description": "Similarity metric: euclidean|cosine|correlation (default: euclidean)"},
        {"name": "scale", "type": "str", "description": "zscore|minmax|none (default: zscore)"},
        {"name": "refine_metric", "type": "str", "description": "dtw|softdtw|affine|ncc|none (default: dtw)"},
        {"name": "search_engine", "type": "str", "description": "ckdtree|hnsw|matrix_profile|mass (default: ckdtree)"}
    ], ["scipy", "numpy"], {"price": True, "return": False, "volatility": False, "ci": True})
