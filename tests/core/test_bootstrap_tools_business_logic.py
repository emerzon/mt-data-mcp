from __future__ import annotations

from types import SimpleNamespace

from mtdata.bootstrap import tools as tools_module


def test_bootstrap_attaches_schemas_only_when_module_set_expands(monkeypatch) -> None:
    first, second = tools_module.TOOL_MODULE_NAMES[:2]
    imported = []
    attached = []

    monkeypatch.setattr(tools_module, "_BOOTSTRAPPED_MODULES", {})
    monkeypatch.setattr(
        tools_module,
        "import_module",
        lambda name: imported.append(name) or SimpleNamespace(__name__=name),
    )
    monkeypatch.setattr(
        tools_module,
        "attach_schemas_to_tools",
        lambda *args: attached.append(args),
    )
    monkeypatch.setattr(tools_module, "get_shared_enum_lists", lambda: {})

    tools_module.bootstrap_tools((first,))
    tools_module.bootstrap_tools((first,))
    tools_module.bootstrap_tools((second,))

    assert imported == [first, second]
    assert len(attached) == 2
