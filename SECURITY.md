# Security

OrdinalDB is an embedded local index. It does not include a network server,
authentication layer, distributed storage, or remote execution surface.

OrdinalDB is 0.2.0 alpha software. Treat `.odb` bundles as untrusted input
unless they come from a source you control. Loads go through
`ordvec-manifest` path, hash, size, and artifact metadata verification before
OrdinalDB parses its own ID sidecar.

Please report security issues privately via a GitHub security advisory:
<https://github.com/Project-Navi/ordinaldb/security/advisories/new>. Do not
open a public issue for suspected vulnerabilities.
