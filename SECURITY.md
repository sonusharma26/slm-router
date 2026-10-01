# Security policy

## Supported version

Security fixes currently target the latest `2.0.0a2` development line. This is an alpha preview, not a hardened multi-tenant gateway.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability** security-advisory form for this repository. Do not include credentials, private prompts, provider responses or customer data in a public issue.

Include the affected version, impact, reproduction conditions and any suggested mitigation. Reports will be acknowledged and assessed before public disclosure.

## Deployment boundary

The service is designed for one trusted operator, one process and one worker. Keep it on a private network or behind an authenticated reverse proxy. Use TLS for remote providers, store credentials outside configuration files, and review [the runtime safety boundaries](docs/release/SAFETY.md) before connecting real traffic.
