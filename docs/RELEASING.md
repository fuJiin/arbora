# Releasing arbora to PyPI

Tag-driven release via GitHub Actions trusted publishing. No API tokens stored anywhere; PyPI verifies the runner via OIDC.

## One-time setup

These steps must be done by a repo + PyPI/TestPyPI admin before the first release.

### 1. Register pending publishers

Both PyPI and TestPyPI support "pending publishers" — you configure the trusted publisher *before* the project exists, and the first successful CI run creates the project.

**TestPyPI** (https://test.pypi.org/manage/account/publishing/):
- PyPI Project Name: `arbora`
- Owner: `fuJiin`
- Repository name: `arbora`
- Workflow name: `release.yml`
- Environment name: `testpypi`

**PyPI** (https://pypi.org/manage/account/publishing/):
- PyPI Project Name: `arbora`
- Owner: `fuJiin`
- Repository name: `arbora`
- Workflow name: `release.yml`
- Environment name: `pypi`

(Same shape both places; just register on each separately. Use the same TestPyPI/PyPI account that should own the published project.)

### 2. Create GitHub environments

In the GitHub repo settings → Environments, create:

- **`testpypi`** — no protection rules needed
- **`pypi`** — recommended: require manual approval before deploying. This adds a "Review pending deployments" prompt before the actual PyPI push, giving a last-second abort. (Settings → Environments → pypi → "Required reviewers" → add yourself.)

### 3. (Optional) Branch protection

The release workflow runs on tag pushes to `main`. If you want to require that the tag commit must have passed CI on `main` first, that's enforced socially today — there's no automated gate. Add one via branch protection if it becomes important.

## Cutting a release

1. Bump `version = "X.Y.Z"` in `pyproject.toml` on `main`. Commit + push.
2. Tag the commit:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```
3. Watch the Actions tab. The pipeline:
   - **build** — runs `uv build`, verifies tag matches `pyproject.toml` version, uploads `sdist` + `wheel` as artifacts
   - **publish-testpypi** — publishes to TestPyPI (always, on every tag)
   - **publish-pypi** — publishes to PyPI (only for stable tags; skipped for `aN`/`bN`/`rcN`/`devN` pre-releases)

   If you set up the `pypi` environment with required reviewers, you'll get a "Review deployment" prompt between TestPyPI and PyPI — approve to continue.

4. Verify:
   - TestPyPI: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arbora==X.Y.Z`
   - PyPI: `pip install arbora==X.Y.Z`

## Pre-release tags

Use PEP 440 markers to publish to TestPyPI only:

- `v0.2.0a1` — alpha
- `v0.2.0b1` — beta
- `v0.2.0rc1` — release candidate
- `v0.2.0.dev1` — dev snapshot

The workflow's `if:` skips the PyPI step for these — useful for sanity-checking the wheel before committing to a real release.

## Yanking a bad release

Releases on PyPI can be yanked (hidden from `pip install` but downloadable by exact pin) via the PyPI web UI: project → Manage → Releases → "Yank". You cannot delete a version; the name + version pair is permanently reserved once published.

For TestPyPI mistakes: same yank flow, lower stakes.

## Troubleshooting

- **"trusted publisher mismatch"** — the workflow filename, environment name, or owner/repo doesn't match what was registered. Re-check the pending publisher config; the strings must match exactly.
- **"version already published"** — bump the version. PyPI versions are immutable.
- **Build fails on `tomllib`** — local repro: `uv run --no-project python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"`.
