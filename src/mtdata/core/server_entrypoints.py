from typing import Callable, Literal, Optional

TransportName = Literal["stdio", "sse", "streamable-http"]


def run_transport_entrypoint(
    main_fn: Callable[..., None],
    *,
    transport: TransportName,
) -> None:
    """Invoke the shared server main with a forced transport."""
    main_fn(transport=transport)


def _make_transport_entrypoint(transport: TransportName) -> Callable[[Callable[..., None]], None]:
    def _entrypoint(main_fn: Callable[..., None]) -> None:
        run_transport_entrypoint(main_fn, transport=transport)

    return _entrypoint


main_stdio = _make_transport_entrypoint("stdio")
main_sse = _make_transport_entrypoint("sse")
main_streamable_http = _make_transport_entrypoint("streamable-http")
