"""Binary PLY with a single vertex element, read and written with numpy.

That is all a Gaussian splat file is: one ``vertex`` element whose properties
are all scalars. Keeping this in-house avoids a GPL dependency for twenty
lines of header parsing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_vertices(path: str | Path) -> np.ndarray:
    """The vertex element of a binary PLY as a structured array."""
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        endian, count, fields, element = None, None, [], None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            words = line.decode("ascii", errors="replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "format":
                endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(words[1])
                if endian is None:
                    raise ValueError(f"{path}: only binary PLY is supported, got {words[1]}")
            elif words[0] == "element":
                element = words[1]
                if element == "vertex":
                    count = int(words[2])
                elif count is not None:
                    raise ValueError(f"{path}: elements after 'vertex' are not supported")
            elif words[0] == "property" and element == "vertex":
                if words[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                fields.append((words[2], endian + _TYPES[words[1]]))
            elif words[0] == "end_header":
                break
        if count is None:
            raise ValueError(f"{path}: no vertex element")
        return np.fromfile(handle, dtype=np.dtype(fields), count=count)


def write_vertices(path: str | Path, data: np.ndarray) -> Path:
    """Write a structured array as the vertex element of a little-endian binary PLY."""
    names = {np.dtype(v).str[1:]: k for k, v in _TYPES.items() if k in
             ("char", "uchar", "short", "ushort", "int", "uint", "float", "double")}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    little = []
    for name in data.dtype.names:
        kind = data.dtype[name].newbyteorder("<")
        header.append(f"property {names[kind.str[1:]]} {name}")
        little.append((name, kind))
    header.append("end_header")
    path = Path(path)
    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        data.astype(np.dtype(little), copy=False).tofile(handle)
    return path
