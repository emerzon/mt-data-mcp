from typing import Callable, Literal, Optional

TransportName = Literal["stdio", "sse", "streamable-http"]


def run_transport_entrypoint(
    main_fn: Callable[..., None],
    *,
    transport: TransportName,
) -> None:
    """Invoke the shared server main with a forced transport."""
    main_fn(transport=transport)


def main_stdio(main_fn: Callable[..., None]) -> None:
    run_transport_entrypoint(main_fn, transport="stdio")


def main_sse(main_fn: Callable[..., None]) -> None:
    run_transport_entrypoint(main_fn, transport="sse")


def main_streamable_http(main_fn: Callable[..., None]) -> None:
    run_transport_entrypoint(main_fn, transport="streamable-http")
