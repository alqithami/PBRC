# Consolidated artifact handoff

This branch is the reviewable core-maintenance portion of the consolidation. It fixes witness sufficiency, duplicate fallback execution, false coverage reporting, independent random generators and safe upstream benchmark checkout. The original main branch remains unchanged until merge.

The full consolidated source-and-results ZIP is supplied separately to the repository owner. It includes the original simulations, the three verified 3000-item KAIROS logs, a maintained table replay, the genuine SciFact source, the complete supplied main200 archive, a portable scorer, manifests and regression tests. It excludes the discarded v0.3.x evidence pilot and private review correspondence.

The large original input and inference archives have NOT been uploaded by this core-maintenance PR. Do not describe this branch alone as the complete experimental artifact. The consolidated ZIP includes scripts/publish_to_github.sh to import the full checked payload into a new review branch using the owner's normal Git credentials, without rewriting main or frozen records.

Canonical original SciFact main200 ZIP SHA-256:
8d2d366df05be1d6ff82a7fcccc01ee80b3da0eaa7c2350334dab3c3c2230b12

Original SciFact v2.3.0 source ZIP SHA-256:
852ca7e13bd4102fc6b28f27c63a971528f4b5da32d3172b41631539f1061b3b

Only main200 has a verified result supplied for this consolidation. full300 is a configuration, not a completed result. The historical KAIROS files on the old public branch have 100 items and are not the submitted 3000-item logs.

The source archive and historical freeze differ in launcher metadata and omitted Ruff caches. The 24 recorded research modules match. Keep maintenance separate from frozen empirical evidence. Hash consistency is not independent proof of scientific validity or past execution.
