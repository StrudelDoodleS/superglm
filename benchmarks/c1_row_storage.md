# C1 row storage and capacity inputs

Packet 2 implements typed, persistent rows with verified bounded reads. Each
payload is at most 8 MiB; the pinned binary index is at most 1 MiB and the
manifest at most 64 KiB. Reads verify metadata before and after reading complete
touched payloads. Returned arrays have immutable byte backing independent of
the store. Reusing cached results still requires fresh source verification.

Independent review approves numbered steps 13–19. The focused suite passes
88 tests, including source corruption, generation changes, immutable ownership,
tail ranges, allocation limits and file-read traffic. Bypassing the payload
digest makes the mutation regression fail.

| Unaligned two-block read | Rows per block | Traced read peak | Payload traffic |
| --- | ---: | ---: | ---: |
| One float64 column | 65,536 | 1,574,385 bytes | 1,048,576 bytes |
| 128 float64 columns | 8,192 | 16,854,252 bytes | 16,777,216 bytes |

Both reads are within the 27,394,048-byte buffer allowance. The pinned metadata
is 441/4,141 bytes for these fixtures. Actual file-read spies also observe two
complete metadata passes. Storage does not claim zero-I/O cache hits or
complete-fit performance.

The [capacity fixture adapter](c1_bounded_fixture.py) generates the existing
fragmented Gaussian fixture by bounded column passes, preserving its seed,
draw order, labels and response arithmetic. It reproduces the eager frame
exactly and the response bitwise at N=37, 257 and 65,537 with different generation
and replay batch boundaries. Its source review found no N-row owner.

| Generated input | Disk payload | Generation wall time | Separately sampled process RAM peak |
| --- | ---: | ---: | ---: |
| 10 million rows | 960 MB | 2.749 s | 126.7 MB |
| 100 million rows | 9.6 GB | 33.791 s | 125.7 MB |

Inputs reside on the project disk. The original 100-million-row run also records
a 639.4 MB process high-water reading; the difference from the later live-RSS
sample is unexplained. Both are retained in the
[receipt](c1_capacity_inputs_receipt.json). Instrumented repeats reproduce all
column hashes, and their clocks are excluded from the timing table. The
10-million-row traced allocation peak is 13.1 MB.

These figures cover input generation and storage. Prepared compilation, bounded
solver endpoints and complete 10-million/100-million fits remain separate C1
acceptance requirements in the [fixed plan](../docs/superpowers/plans/2026-09-11-c1-closeout.md).
