"""Central runtime entrypoint: orchestrates engine runner only."""

from __future__ import annotations

import asyncio

from schism.runtime.engine_runner import run_forever


async def main() -> None:
    """Run the model engine process."""
    await run_forever()


if __name__ == "__main__":
    asyncio.run(main())
