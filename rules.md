# rules.md

No AI slop.

## Do

- One function = one job
- Delete unused code
- Fail fast with asserts
- One kernel per file
- Fuse ops that share data

## Don't

- Abstractions until third use
- Wrappers that add nothing
- Comments that restate code
- Classes for single functions
- Type hints that mirror names
- Logging in hot paths
