"""PagedServe benchmark harness.

Deliberately independent of the engine: nothing here may import ``pagedserve``
or ``torch`` at module scope, so the harness runs on a laptop with no GPU and
can drive a mock backend, an in-process engine, or a remote HTTP endpoint
through the same interface.
"""

from __future__ import annotations
