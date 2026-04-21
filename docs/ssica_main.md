# `SSICA/main.py`

## Role

Runs the queue-based SSICA simulation loop.

## Main Differences From SSIA

- tasks can be queued per agent rather than assigned one at a time with no look-ahead
- queue activation and invalidation logic is more stateful than SSIA

## Refactor-Relevant Behavior

Blocked-path handling now routes through the auctioneer instead of directly marking the task complete or dropping it in the loop.

That keeps the “repair first, then reauction if needed” rule centralized even though SSICA has more queue-specific state to maintain.
