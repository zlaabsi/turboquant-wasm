# Contributing

## Setup

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

git clone https://github.com/zlaabsi/turboquant-wasm.git
cd turboquant-wasm
```

`npm install` is optional for day-to-day Rust and WASM work because the repo does not require a bundler to build the examples.

## Local Workflow

```bash
npm run build
npm run build:node
npm run test
npm run verify
```

To run the examples:

```bash
npm run build
python3 -m http.server 8080
```

## Commit Convention

Use Conventional Commits:

- `feat:` new features or new examples
- `fix:` bug fixes
- `docs:` documentation and cookbook updates
- `perf:` measurable performance work
- `refactor:` structural changes without behavior change
- `test:` test-only changes
- `chore:` maintenance and tooling

Examples:

- `feat: add indexeddb cookbook for client-side rag`
- `fix: reject mismatched quantizers in search`
- `docs: add edge deployment guide`

## Pull Request Expectations

- Keep Rust changes covered by `wasm-bindgen` tests when behavior changes.
- Do not commit generated `pkg/`, `pkg-node/`, or `target/`.
- Prefer example additions over abstract documentation when introducing a new use case.
- Document any API changes in `README.md` and the relevant cookbook page.
