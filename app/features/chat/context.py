"""Prior-context lookup for a client.

Stub implementation returns empty payloads so the graph runs without a
database. Wire ContextRepo to your actual store (Postgres, Firestore,
whatever) by overriding fetch_for_client.
"""

from __future__ import annotations

from typing import Any, Protocol


class ClientContext(dict[str, Any]):
    """Same as a dict but easier to grep for at call sites."""


class ContextSource(Protocol):
    async def fetch_for_client(self, client_id: str) -> ClientContext: ...


class ContextRepo:
    """Default no-op source. Returns the shape downstream nodes expect."""

    async def fetch_for_client(self, client_id: str) -> ClientContext:
        return ClientContext(
            prior_insights=[],
            open_case_data={},
            last_sentiment=None,
            last_risk=None,
        )


# module-level singleton — swap with monkeypatch in tests / DI in prod
default_repo: ContextSource = ContextRepo()
