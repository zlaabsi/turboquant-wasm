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

## Release Flow

The repository release tag is `v<version>`, for example `v0.1.0`.
The package is published as `@zlaabsi/turboquant-wasm` on npmjs and mirrored to GitHub Packages.

Normal release flow:

1. Update `package.json` version.
2. Run `npm run verify`.
3. Push the release commit to `main`.
4. Create and push the matching git tag.

The release workflow validates the tag, runs checks, attaches the tarball to the GitHub Release, publishes to npm through npm trusted publishing from GitHub Actions, then publishes the same tarball to GitHub Packages with the repository `GITHUB_TOKEN`. No `NPM_TOKEN` repository secret is required.

If a version already exists on npm but has not been mirrored to GitHub Packages yet, run the `Sync GitHub Packages` workflow from `main`. It republishes the current tarball to `npm.pkg.github.com` without touching npmjs.

Before the first automated publish, configure the npm package to trust this repository and workflow:

1. Open the npm package settings for `@zlaabsi/turboquant-wasm`.
2. Add a trusted publisher for GitHub Actions.
3. Use repository `zlaabsi/turboquant-wasm` and workflow filename `release.yml`.

After that one-time setup, pushing `v<version>` is enough to publish.

After the first GitHub Packages publish, check the package visibility in GitHub if you want the package page visible outside maintainers. GitHub Packages npm packages start private by default, even when linked to a repository.

## Pull Request Expectations

- Keep Rust changes covered by `wasm-bindgen` tests when behavior changes.
- Do not commit generated `pkg/`, `pkg-node/`, or `target/`.
- Prefer example additions over abstract documentation when introducing a new use case.
- Document any API changes in `README.md` and the relevant cookbook page.
