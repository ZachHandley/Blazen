"""Stateful workflow example.

Demonstrates the two explicit Context namespaces and the
identity-preserving ``StopEvent.result`` feature:

1. ``ctx.state``  -- persistable values that survive ``pause()`` /
   ``resume()``.  Use for counters, JSON-friendly snapshots, anything
   you would want serialised into a checkpoint.
2. ``ctx.session`` -- live in-process references (DB connections, open
   files, lambdas, ML models).  **Identity-preserving** within a run:
   ``ctx.session["conn"]`` returns the *same* Python object across
   steps.  Excluded from snapshots.
3. ``StopEvent(result=obj)`` -- the returned object's ``is``-identity
   is preserved end-to-end, so the caller receives the exact same
   Python instance the workflow produced.

Flow::

    StartEvent -> setup  -> QueryEvent  (writes ctx.state + ctx.session)
    QueryEvent -> query  -> StopEvent   (reads both back, returns conn)

Run with: python stateful_workflow.py
"""

import asyncio
import sqlite3

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# Step 1: Open a sqlite connection, seed it, and hand off.
#
# - ``ctx.state`` holds a persistable counter (would survive a snapshot).
# - ``ctx.session`` holds the live connection (excluded from snapshots,
#   identity preserved across steps).
# ---------------------------------------------------------------------------
@step
async def setup(ctx: Context, ev: StartEvent) -> Event:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO users (name) VALUES (?)",
        [("alice",), ("bob",), ("carol",)],
    )
    conn.commit()

    # Persistable state -- safe to snapshot.
    ctx.state["row_count_expected"] = 3

    # Live reference -- identity-preserving, not snapshotted.
    ctx.session["db"] = conn

    # Identity is preserved on reads within the same step.
    assert ctx.session["db"] is conn

    print(f"  [setup] stored 3 users; conn id={id(conn)}")
    return Event("QueryEvent")


# ---------------------------------------------------------------------------
# Step 2: Read the live connection back from session and query it.
#
# Demonstrates cross-step identity preservation: the ``conn`` we get
# here is the *same* Python object that ``setup`` stored.
# ---------------------------------------------------------------------------
@step(accepts=["QueryEvent"])
async def query(ctx: Context, ev: Event) -> StopEvent:
    conn = ctx.session["db"]
    (actual_count,) = conn.execute("SELECT count(*) FROM users").fetchone()

    # Update persistable state with the observed count.
    ctx.state["row_count_actual"] = actual_count

    print(f"  [query]  counted {actual_count} users; same conn? {conn is ctx.session['db']}")

    # Return the live connection via StopEvent.result -- identity is
    # preserved end-to-end so the caller receives the exact same object.
    return StopEvent(result=conn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    wf = Workflow("stateful-example", [setup, query])

    handler = await wf.run()
    result = await handler.result()

    returned_conn = result.result
    print(f"  [final]  result is sqlite3.Connection? {isinstance(returned_conn, sqlite3.Connection)}")

    # The returned connection is still live -- keep using it.
    (alice_row,) = returned_conn.execute(
        "SELECT name FROM users WHERE id = 1"
    ).fetchone()
    print(f"  [final]  alice row -> {alice_row}")

    returned_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
