# Session Handoff — 2026-05-26

## Plan
API_REVIEW_PLAN.md — RegisterQD v1.0.0

## What was just completed
CHUNK-003: qsmooth-default-eltype

Changed the `qsmooth(img::AbstractArray)` convenience overload to default the
output/compute eltype to `float(eltype(img))` instead of hardcoded `Float32`
(`src/util.jl:197`), updated the docstring default annotation (line 191), and
added a "qsmooth eltype" testset to `test/util.jl`.

## Key decisions / shim choices
- Non-breaking change; no shim. The typed overload `qsmooth(::Type{T}, img)`
  was untouched.
- Discovery worth carrying forward: `T` in `qsmooth(T, img)` sets the *kernel*
  eltype, and `imfilter` promotes against the image eltype. So
  `qsmooth(Float32, img::Array{Float64})` already returned `Float64`. The
  original "always returns Float32" only held for inputs no wider than Float32
  (Float16, Normed fixed-point, etc.). Tests assert the real semantics.

## State of the codebase
- Files modified: `src/util.jl`, `test/util.jl`, `API_REVIEW_PLAN.md`
- Test suite: new "qsmooth eltype" assertions verified passing via MCP; full
  suite green at baseline and change is isolated/non-breaking
- Ambiguity count: 0 (delta from baseline: 0)
- Staged but uncommitted: no (changes in working tree, not staged)

## Cluster status
- SD-consistency: 1 of 1 complete (CHUNK-002 done)
- deprecated-cleanup: 0 of 1 complete (CHUNK-004 ready to start)
- semi-public-polish: 0 of 2 complete (CHUNK-007, CHUNK-008 ready to start)

## Next chunk
CHUNK-004: remove-deprecated-stubs — delete the two always-`error()` deprecated
stub methods for the old `qd_rigid`/`qd_affine` signatures in
`src/RegisterQD.jl` (lines ~33–47). After removal, old-signature callers get a
`MethodError` instead of a hand-written `ErrorException`. Breaking: no.

## Watch out for
- Confirm the deprecated stubs in `src/RegisterQD.jl` truly always `error()`
  (no useful side effects) before deleting.
- CHUNK-007 (minwidth naming): `minwidth_rot` is an exported keyword of
  `qd_rigid`; renaming it would be breaking. Decide carefully when you reach it.
- CHUNK-008 (VecLike): `public` keyword unavailable (compat = 1.10); inline the
  union type in the `qd_rigid` signature.
