"""The systemd-scope vocabulary, declared once.

`ollama.service` lives in exactly one systemd manager: the system one or the
caller's user one. Both helper scripts address it by unit name plus a scope,
the scope arrives on each script's command line as `--systemd-scope`, and the
rule for turning a scope into a systemctl invocation is identical in both.
Before this module the constant pair, the flag, the error string and the
`--user` rule were hand-copied across the two scripts -- CONTRIBUTING.md trap
#12: values duplicated across a boundary fail silently on a one-sided edit,
and here the failure mode was the collector and the action helper disagreeing
about which unit they address with a fully green suite. Declared once, here.

No imports, and no I/O: the scripts stay stdlib-only, and this keeps the
shared rule as boring as it looks.
"""

SYSTEMD_SCOPES = ("system", "user")
DEFAULT_SYSTEMD_SCOPE = "system"
SCOPE_FLAG = "--systemd-scope"
SCOPE_ERROR = SCOPE_FLAG + " must be system or user"


def normalize_scope(value):
    """Fold a scope name to its canonical spelling, or raise ValueError.

    Case is folded because the manifest's menu labels are "System" and
    "User", and a hand-edited shell.json will not agree with them about
    capitalisation -- `"User" == "user"` failing used to make a User-scope
    caller address the SYSTEM unit (polkit prompt and all) with no error at
    all. Anything that is not one of the two names raises rather than
    defaulting to system: defaulting acts on a unit nobody selected.
    Service.qml sanitises before spawning, so the widget cannot send a value
    this rejects; raising is for direct callers, tests, and any future script.
    """
    if not isinstance(value, str):
        raise ValueError(SCOPE_ERROR + " (got " + type(value).__name__ + ")")
    scope = value.strip().lower()
    if scope not in SYSTEMD_SCOPES:
        raise ValueError(SCOPE_ERROR + " (got " + repr(value) + ")")
    return scope


def systemctl_scope_args(scope):
    """The systemctl flags selecting the manager for this scope.

    The single home of the `--user` rule: both scripts assemble their
    systemctl command through this. Unknown scopes raise rather than being
    handed the system command -- a caller that skipped validation still must
    not act on the unit it did not name.
    """
    return ["--user"] if normalize_scope(scope) == "user" else []
