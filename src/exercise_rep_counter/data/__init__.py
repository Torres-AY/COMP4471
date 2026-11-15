"""Dataset discovery, manifests, and subject-safe splitting utilities."""

from .manifest import SampleRecord, read_manifest, write_manifest

__all__ = ["SampleRecord", "read_manifest", "write_manifest"]
