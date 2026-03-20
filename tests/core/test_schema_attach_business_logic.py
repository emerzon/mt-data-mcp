from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from mtdata.core import schema_attach as schema_attach_mod


def _attach_tool_schema(
    monkeypatch, tool_name: str, base_schema: dict, *, shared_enums: dict | None = None
):
    def tool_func():
        return None

    tool_func.__name__ = tool_name
    tool_obj = SimpleNamespace(func=tool_func)
    apply_calls: list[dict] = []

    monkeypatch.setattr(
        schema_attach_mod, "get_mcp_registry", lambda _mcp: {tool_name: tool_obj}
    )
    monkeypatch.setattr(
        schema_attach_mod,
        "_get_function_info",
        lambda func: {"name": tool_name, "parameters": []},
    )
    monkeypatch.setattr(
        schema_attach_mod, "_build_minimal_schema", lambda info: deepcopy(base_schema)
    )
    monkeypatch.setattr(
        schema_attach_mod,
        "_enrich_schema_with_shared_defs",
        lambda schema, info: schema,
    )
    monkeypatch.setattr(
        schema_attach_mod,
        "_complex_defs",
        lambda: {
            "IndicatorSpec": {"type": "object"},
            "DenoiseSpec": {"type": "object"},
            "SimplifySpec": {"type": "object"},
        },
    )
    monkeypatch.setattr(
        schema_attach_mod,
        "_apply_param_hints",
        lambda schema: apply_calls.append(deepcopy(schema)),
    )

    schema_attach_mod.attach_schemas_to_tools(object(), shared_enums or {})
    return tool_obj, tool_func, apply_calls


def test_attach_schemas_to_tools_patches_forecast_generate(monkeypatch) -> None:
    tool_obj, tool_func, apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_generate",
        {
            "parameters": {
                "properties": {
                    "quantity": {"type": "string"},
                    "denoise": {"type": "object"},
                    "params": {"type": "string"},
                },
                "required": ["quantity"],
            }
        },
    )

    schema = tool_obj.schema
    params = schema["parameters"]["properties"]
    assert params["quantity"] == {"$ref": "#/$defs/QuantitySpec"}
    assert params["denoise"] == {
        "anyOf": [{"$ref": "#/$defs/DenoiseSpec"}, {"type": "null"}]
    }
    assert params["params"] == {"type": "object", "additionalProperties": True}
    assert tool_func.schema == schema
    assert len(apply_calls) == 1


def test_attach_schemas_to_tools_patches_indicator_and_data_refs(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "data_fetch_candles",
        {
            "parameters": {
                "properties": {
                    "indicators": {"type": "string"},
                    "denoise": {"type": "object"},
                    "simplify": {"type": "object"},
                },
                "required": [],
            }
        },
    )

    params = tool_obj.schema["parameters"]["properties"]
    indicator_any_of = params["indicators"]["anyOf"]
    assert {
        "type": "array",
        "items": {"$ref": "#/$defs/IndicatorSpec"},
    } in indicator_any_of
    assert any(option.get("type") == "string" for option in indicator_any_of)
    assert {"type": "null"} in indicator_any_of
    assert params["denoise"] == {
        "anyOf": [{"$ref": "#/$defs/DenoiseSpec"}, {"type": "null"}]
    }
    assert params["simplify"] == {
        "anyOf": [{"$ref": "#/$defs/SimplifySpec"}, {"type": "null"}]
    }

    indicator_obj, _indicator_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "indicators_list",
        {
            "parameters": {
                "properties": {
                    "category": {"type": "string"},
                },
                "required": [],
            }
        },
        shared_enums={"CATEGORY_CHOICES": ["trend", "momentum"]},
    )

    indicator_params = indicator_obj.schema["parameters"]["properties"]
    assert indicator_params["category"] == {"$ref": "#/$defs/IndicatorCategory"}


def test_attach_schemas_to_tools_patches_barrier_method_enums(monkeypatch) -> None:
    prob_obj, _prob_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_barrier_prob",
        {
            "parameters": {
                "properties": {
                    "method": {"type": "string"},
                },
                "required": [],
            }
        },
    )
    prob_method = prob_obj.schema["parameters"]["properties"]["method"]
    assert "closed_form" in prob_method["enum"]
    assert "auto" in prob_method["enum"]

    opt_obj, _opt_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "forecast_barrier_optimize",
        {
            "parameters": {
                "properties": {
                    "method": {"type": "string"},
                },
                "required": [],
            }
        },
    )
    opt_method = opt_obj.schema["parameters"]["properties"]["method"]
    assert "closed_form" not in opt_method["enum"]
    assert "auto" in opt_method["enum"]


def test_attach_schemas_to_tools_patches_trade_place(monkeypatch) -> None:
    tool_obj, _tool_func, _apply_calls = _attach_tool_schema(
        monkeypatch,
        "trade_place",
        {
            "parameters": {
                "properties": {
                    "order_type": {"type": "string"},
                    "expiration": {"type": "string"},
                },
                "required": [],
            }
        },
    )

    schema = tool_obj.schema
    params_obj = schema["parameters"]
    params = params_obj["properties"]

    assert params_obj["required"] == ["symbol", "volume", "order_type"]
    assert params["order_type"]["anyOf"][0]["enum"][0] == "BUY"
    assert params["order_type"]["anyOf"][1]["enum"] == [0, 1, 2, 3, 4, 5]
    assert params["expiration"]["anyOf"] == [
        {"type": "string"},
        {"type": "number"},
        {"type": "null"},
    ]
