# Vendored third-party code: TradingView Lightweight Charts™

| | |
|---|---|
| **Package** | `lightweight-charts` |
| **Version** | **5.2.1** (pinned; see "Upgrading" below) |
| **File** | `lightweight-charts.standalone.production.js` |
| **Source** | `https://cdn.jsdelivr.net/npm/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js` |
| **Size** | 197,922 bytes |
| **SHA-256** | `e21cc5caa0226ef30bd8549c50b9ef926615f2a4ee6b4e486353477a55f598cf` |
| **Licence** | Apache License 2.0 — full text in `LICENSE-lightweight-charts.txt` (SHA-256 `70c9d5382506dd184465425c08a99ad9bd6d9ac1313c252968ba0b585e5ef823`) |
| **Copyright** | Copyright (c) 2026 TradingView, Inc. |
| **Modified?** | **No.** Vendored byte-for-byte as published. |

## Why it is vendored rather than loaded from a CDN

This file is served by the monitoring dashboard, which runs inside a live-money
trading process. Pulling remote JavaScript into a page that displays open
positions would put a third party in the trust path of a page the operator
reads to make decisions, and would make the dashboard depend on the internet
being up during a session. The page's Content-Security-Policy is
`script-src 'self'`, so nothing off-box can execute there even by accident.

The SHA-256 above is the point of this file: it lets any later reader prove
the bundle was never hand-edited. Verify it with:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('Dependencies/dashboard_assets/vendor/lightweight-charts.standalone.production.js').read_bytes()).hexdigest())"
```

## Attribution

The library's built-in attribution logo is deliberately left enabled — do not
pass `layout.attributionLogo: false` in `dashboard.js`. The dashboard also
carries a visible footer credit.

## Upgrading

The v4 and v5 APIs differ where this page touches them: v5 is
`chart.addSeries(LightweightCharts.CandlestickSeries, {...})`, v4 was
`chart.addCandlestickSeries({...})`. On any version change, update
`dashboard.js` in the same commit, and update the version, size and SHA-256 in
the table above.
