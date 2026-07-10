# Local agent instructions

## App server

- Keep only one Voice Light app server instance running.
- Before starting the app, check for an existing Voice Light listener and restart that instance instead of starting the app on a second port.
- Prefer port `8000` for the local app unless the user explicitly requests another port.
