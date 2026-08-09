import subprocess
import sys
import textwrap
from pathlib import Path


def test_pattern_imports_defer_optional_numerical_backends():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(src)!r})
        sys.path.insert(0, {str(root)!r})

        import mtdata.core.patterns

        assert "mtdata.utils.patterns" not in sys.modules
        assert "stumpy" not in sys.modules
        assert "sklearn" not in sys.modules
        assert "umap" not in sys.modules

        import mtdata.utils.patterns
        from mtdata.utils.dimred import list_dimred_methods

        methods = list_dimred_methods()
        assert isinstance(methods["pca"]["available"], bool)
        assert "stumpy" not in sys.modules
        assert "sklearn" not in sys.modules
        assert "umap" not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
