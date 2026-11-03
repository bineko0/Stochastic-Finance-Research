"""Data sources. Importing this package registers every built-in source.

To add a source (e.g. Bloomberg via blpapi): create a module here with a
`DataSource` subclass decorated with @SOURCES.register("name"), import it below,
and add its parameters under `sources:` in configs/data.yaml.
"""
from sfr.data.sources import csv_source, synthetic, yahoo  # noqa: F401
from sfr.data.sources.base import DataSource

__all__ = ["DataSource"]
