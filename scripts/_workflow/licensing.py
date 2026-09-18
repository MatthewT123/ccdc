"""Load only CCDC licensing configuration from the repository's private .env."""

import os
from pathlib import Path
import shlex

VARIABLE = "CCDC_LICENSING_CONFIGURATION"
DEFAULT_ENV = Path(__file__).resolve().parents[2] / ".env"


def load_ccdc_license(path=DEFAULT_ENV):
    """Respect exported configuration; never execute or interpolate .env content."""
    if VARIABLE in os.environ:
        return
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if not separator or name.strip() != VARIABLE:
            continue
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise ValueError("Invalid CCDC licensing value in .env; check quoting") from None
        if len(parts) != 1 or not parts[0].startswith(("la-code;", "lf-server;")) or not parts[0].split(";", 1)[1]:
            raise ValueError("Set CCDC_LICENSING_CONFIGURATION in .env to 'la-code;YOUR_ACTIVATION_KEY' or 'lf-server;URL'")
        if parts[0].endswith(";YOUR_ACTIVATION_KEY"):
            raise ValueError("Replace YOUR_ACTIVATION_KEY in .env with your private CCDC key")
        os.environ[VARIABLE] = parts[0]
        return
