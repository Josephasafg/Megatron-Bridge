import io

import pytest
from megatron.core.msc_utils import MultiStorageClientFeature


class _RecordingWriter(io.BytesIO):
    """Write-mode handle: commits its bytes into the fake's blob store on close."""

    def __init__(self, msc: "RecordingMsc", path: str, binary: bool) -> None:
        super().__init__()
        self._msc = msc
        self._path = path
        self._binary = binary

    def write(self, data) -> int:
        if not self._binary and isinstance(data, str):
            data = data.encode()
        return super().write(data)

    def close(self) -> None:
        self._msc.blobs[self._path] = self.getvalue()
        super().close()


class RecordingMsc:
    """Fake multistorageclient package: an in-memory blob store behind ``msc.open``."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def open(self, path: str, mode: str = "r"):  # noqa: A003 — mirrors msc.open
        binary = "b" in mode
        if "r" in mode:
            if path not in self.blobs:
                raise FileNotFoundError(path)
            data = self.blobs[path]
            return io.BytesIO(data) if binary else io.StringIO(data.decode())
        return _RecordingWriter(self, path, binary)


@pytest.fixture
def fake_msc(monkeypatch):
    """Turn the MSC feature flag on and back it with an in-memory package fake."""
    msc = RecordingMsc()
    # MultiStorageClientFeature is a module-level instance, so plain callables replace its bound methods.
    monkeypatch.setattr(MultiStorageClientFeature, "is_enabled", lambda: True)
    monkeypatch.setattr(MultiStorageClientFeature, "import_package", lambda: msc)
    return msc
