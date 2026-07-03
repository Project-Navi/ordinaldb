# Provenance

OrdinalDB was bootstrapped in early development from the MIT-licensed
Turbovec repository (upstream commit
`15694dd0e8672c9664e521f05dd12a453c094a10`), then narrowed around
OrdVec-backed ordinal retrieval and embedded adapter persistence.

All Turbovec-derived code and prose have since been replaced with
independent implementations, verified by a file-level provenance audit.
OrdinalDB is not a fork or distribution of Turbovec; the lineage
acknowledgment lives in `NOTICE` and `THIRD_PARTY.md`.

Historical algorithm terms that may appear in provenance records or old
changelog entries — TurboQuant, TQ+, Lloyd-Max, codebook, centroid,
rotation matrix, learned rotation, random rotation, product quantization,
scalar quantization, FAISS, Shannon — are bootstrap-era references only,
not OrdinalDB roadmap items. The algorithmic substrate is ordinal
retrieval through OrdVec.
