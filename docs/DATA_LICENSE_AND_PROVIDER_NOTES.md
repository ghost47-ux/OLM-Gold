# Data License and Provider Notes

This is an engineering note, not legal advice. Verify the current terms with Twelve Data and, where relevant, with the account owner before ingesting, retaining, training on, or redistributing any data.

## What this repository contains

- **No real market data.** The only bundled/generated datasets are deterministic synthetic fixtures, clearly labeled `source=synthetic` and written to `XAUUSD_synthetic_*` paths.
- **No API keys.** `.env.example` contains empty placeholders. `.env` is git-ignored and must never be committed, copied into documentation, or pasted into an issue.
- **Code only.** A local reminder of the data terms is not a license to redistribute provider data.

## Provider identity

| Field | Expected value |
| --- | --- |
| Provider | Twelve Data |
| Symbol | `XAU/USD` |
| Exchange/venue | `Commodity Aggregate` |
| Timezone | UTC |
| Bars | M1, M5, M15, H1, H4 |

The ingestion probe records the `meta.exchange` returned by the provider. Do not silently substitute `XAUUSD` from another venue, futures, ETFs or a different gold feed. A different instrument has different microstructure, sessions and roll/contract behavior.

## Authorization gates

All gates default to false:

| Environment variable | Controls |
| --- | --- |
| `TD_MULTI_ACCOUNT_POOL_AUTHORIZED` | Whether multiple accounts may be pooled into one scheduler/pool |
| `TD_MODEL_TRAINING_APPROVED` | Whether provider-sourced data may be used for training |
| `TD_DATA_RETENTION_APPROVED` | Whether historical data may be retained on disk |
| `TD_COMMERCIAL_USE_APPROVED` | Whether outputs/data may be used commercially |

Bulk retention (`download-history` without `--dry-run`) and provider-data training check the relevant gate in code. `--dry-run` is network-free and gate-free so a user can inspect request volume before making a contractual decision.

Enabling a gate is a statement by the operator that permission exists. The software cannot verify a contract, account ownership, plan limits, or redistribution rights.

## Keys and account safety

- Read keys only from environment variables (`TWELVE_DATA_API_KEY_1..7`). Never hardcode them in source, YAML, notebooks or tests.
- Keys shown in chat, logs, screenshots or shell history should be treated as exposed. Rotate them with the provider and use the replacements only in `.env`.
- The CLI masks credentials as first four plus last two characters in human reports. It never prints the full key.
- Log redaction masks API-key query parameters, bearer tokens, known environment secrets and 32-character hex strings. This is defense in depth, not a guarantee that every future secret format is covered.
- Per-account minute/daily budgets, cooldowns and a safety margin limit request bursts. `olm usage` reports counts and errors without keys.
- Multi-account pooling is off by default because it may violate plan terms or account ownership even when the keys are technically usable.

## Retention and redistribution

Unless the provider explicitly grants it:

- Do not commit raw or staged provider Parquet files to git.
- Do not publish derived datasets that reconstruct provider history.
- Do not include provider data in a public model release or demo.
- Keep retention limited to the stated research purpose and delete it when that purpose ends.

The project writes provider data under `<data-root>/raw` and `<data-root>/staged`, and ignores those Parquet files by default. Verify `.gitignore` in your fork before committing.

## Commercial use

`TD_COMMERCIAL_USE_APPROVED` defaults to false. A model trained on provider data may still be subject to the provider terms even if the weights contain no literal rows. Do not enable this gate or ship a commercial service without an actual license review.

## Synthetic data

Synthetic fixtures are generated locally from a seeded process. They are useful for tests, CI, offline development and pipeline demonstrations. They are **not** a substitute for market history, and any metric computed on them must be labeled synthetic. Never present a synthetic result as evidence about gold.

## Attribution and citation

If you publish work using Twelve Data, follow their current attribution requirements. Cite the provider, instrument, venue, interval, retrieval date, timezone and the dataset hash recorded in the prepared manifest. Do not cite this repository as the source of the prices.

## Incident checklist

1. Revoke/rotate the exposed key at the provider.
2. Remove the secret from the working tree and, if committed, from git history.
3. Check `olm usage` and provider dashboards for unexpected requests.
4. Re-run `verify-provider` with the new key and confirm the masked label and instrument metadata.
5. Do not paste the replacement key into chat or a public issue.
