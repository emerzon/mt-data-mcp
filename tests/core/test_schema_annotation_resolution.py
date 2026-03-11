from __future__ import annotations

from typing import get_args, get_origin
from typing_extensions import TypedDict

from mtdata.core import cli
from mtdata.core.schema import get_function_info


class ExampleSpec(TypedDict, total=False):
    method: str
    points: int


def annotated_tool(
    count: int | None = None,
    enabled: bool | None = None,
    spec: ExampleSpec | None = None,
) -> dict[str, object]:
    return {
        "count": count,
        "enabled": enabled,
        "spec": spec,
    }


def test_get_function_info_resolves_future_annotations():
    info = get_function_info(annotated_tool)
    params = {p["name"]: p for p in info["params"]}

    count_type = params["count"]["type"]
    enabled_type = params["enabled"]["type"]
    spec_type = params["spec"]["type"]

    assert get_origin(count_type) in (cli.Union, cli.types.UnionType)
    assert int in get_args(count_type)
    assert type(None) in get_args(count_type)

    assert get_origin(enabled_type) in (cli.Union, cli.types.UnionType)
    assert bool in get_args(enabled_type)
    assert type(None) in get_args(enabled_type)

    base_type, _ = cli._unwrap_optional_type(spec_type)
    kwargs, is_mapping = cli._resolve_param_kwargs(params["spec"], None)

    assert base_type is ExampleSpec
    assert is_mapping is True
    assert kwargs["type"] is str
