#!/usr/bin/env python3
"""Compatibility wrapper for the packaged Web API entrypoint."""


def main():
    from src.mtdata.core.web_api import main_webapi

    main_webapi()


if __name__ == "__main__":
    main()
