"""The 8 mutation operators (PRD section 5, "Action space").

Every operator has the signature ``op(data, rng, pool) -> bytes``:
  - data: the seed being mutated (bytes)
  - rng:  a random.Random instance, so runs are reproducible from a seed
  - pool: list of bytes, the current seed pool (only splice uses it)

Every operator always returns non-empty bytes, even when given empty input.

Python 3.5 compatible on purpose: this module runs inside the ARVO
containers, which ship Python 3.5.2 (see feasibility_12950.log).
No f-strings, no variable annotations.
"""

import struct

NUM_ACTIONS = 8

# Boundary values from the PRD, grouped by the width (bytes) they fit in.
INTERESTING_VALUES = [0, 1, 0x7F, 0x80, 0xFF, 0x7FFF, 0xFFFF,
                      0x7FFFFFFF, 0xFFFFFFFF]
_WIDTH_FORMAT = {1: "B", 2: "H", 4: "I"}
_WIDTH_MAX = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}


def _ensure_nonempty(data, rng):
    # NOTE: an empty seed has nothing to mutate; we start from one random
    # byte so every operator can still return valid, non-empty output.
    if len(data) == 0:
        return bytes([rng.randrange(256)])
    return data


def bit_flip(data, rng, pool):
    """a1: flip one random bit."""
    buf = bytearray(_ensure_nonempty(data, rng))
    pos = rng.randrange(len(buf))
    buf[pos] ^= 1 << rng.randrange(8)
    return bytes(buf)


def byte_flip(data, rng, pool):
    """a2: XOR one random byte with 0xFF."""
    buf = bytearray(_ensure_nonempty(data, rng))
    buf[rng.randrange(len(buf))] ^= 0xFF
    return bytes(buf)


def random_byte(data, rng, pool):
    """a3: replace one byte with a random value."""
    buf = bytearray(_ensure_nonempty(data, rng))
    pos = rng.randrange(len(buf))
    old = buf[pos]
    # NOTE: draw from the 255 values that differ from the current byte, so
    # the operator always changes the input.
    new = rng.randrange(255)
    buf[pos] = new if new < old else new + 1
    return bytes(buf)


def insert_bytes(data, rng, pool):
    """a4: insert 1-4 random bytes at a random position."""
    buf = bytearray(data)
    n = rng.randint(1, 4)
    pos = rng.randint(0, len(buf))
    buf[pos:pos] = bytes(rng.randrange(256) for _ in range(n))
    return bytes(buf)


def delete_bytes(data, rng, pool):
    """a5: delete 1-4 bytes at a random position."""
    buf = bytearray(_ensure_nonempty(data, rng))
    # NOTE: never delete the last byte; the output must stay non-empty.
    if len(buf) <= 1:
        return bytes(buf)
    n = rng.randint(1, min(4, len(buf) - 1))
    pos = rng.randint(0, len(buf) - n)
    del buf[pos:pos + n]
    return bytes(buf)


def interesting_value(data, rng, pool):
    """a6: overwrite 1/2/4 bytes with a boundary value."""
    buf = bytearray(_ensure_nonempty(data, rng))
    widths = [w for w in (1, 2, 4) if w <= len(buf)]
    width = rng.choice(widths)
    candidates = [v for v in INTERESTING_VALUES if v <= _WIDTH_MAX[width]]
    value = rng.choice(candidates)
    # NOTE: byte order is chosen at random; the PRD does not specify one and
    # targets differ (PNG is big-endian, PE is little-endian).
    endian = rng.choice(("<", ">"))
    packed = struct.pack(endian + _WIDTH_FORMAT[width], value)
    pos = rng.randint(0, len(buf) - width)
    buf[pos:pos + width] = packed
    return bytes(buf)


def duplicate_chunk(data, rng, pool):
    """a7: copy a 1-32 byte chunk to a random position (inserted, not overwritten)."""
    buf = bytearray(_ensure_nonempty(data, rng))
    n = rng.randint(1, min(32, len(buf)))
    src = rng.randint(0, len(buf) - n)
    chunk = bytes(buf[src:src + n])
    dst = rng.randint(0, len(buf))
    buf[dst:dst] = chunk
    return bytes(buf)


def splice(data, rng, pool):
    """a8: join a prefix of the input with a suffix of another pool seed."""
    data = _ensure_nonempty(data, rng)
    others = [s for s in (pool or []) if len(s) > 0 and s != data]
    if not others:
        # NOTE: no other seed to splice with (e.g. pool of one); fall back to
        # splicing the input with itself at two different cut points.
        other = data
    else:
        other = rng.choice(others)
    cut_a = rng.randint(1, len(data))        # prefix keeps >= 1 byte
    cut_b = rng.randint(0, len(other) - 1)   # suffix keeps >= 1 byte
    return data[:cut_a] + other[cut_b:]


# Index i is action a(i+1) in the PRD table.
OPERATORS = [
    bit_flip,
    byte_flip,
    random_byte,
    insert_bytes,
    delete_bytes,
    interesting_value,
    duplicate_chunk,
    splice,
]

OPERATOR_NAMES = [op.__name__ for op in OPERATORS]


def mutate(action, data, rng, pool=None):
    """Apply operator number ``action`` (0-7) to ``data``."""
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError("action must be in 0..7, got %r" % (action,))
    return OPERATORS[action](bytes(data), rng, pool or [])
