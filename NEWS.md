# version 1.0.0

## Breaking changes

- `rotation_gridsearch`: the `SD` argument is now a keyword argument. Calls that
  passed `SD` positionally (`rotation_gridsearch(fixed, moving, maxshift, maxradians,
  rgridsz, mySD)`) must be updated to keyword syntax
  (`rotation_gridsearch(fixed, moving, maxshift, maxradians, rgridsz; SD=mySD)`).

## Non-breaking changes

- `qsmooth(img)` now infers the output element type from the input: `float(eltype(img))`
  instead of always returning `Float32`. To keep `Float32` output, pass it explicitly:
  `qsmooth(Float32, img)`.
- `default_minrot` now accepts an `AbstractArray` directly in addition to
  `CartesianIndices`.
- The old `qd_rigid` and `qd_affine` deprecated stubs (which always threw an error)
  have been removed; mismatched calls now produce a clearer `MethodError`.

---

# version 0.3

## Breaking changes

- RegisterQD now requires an extra step because of enhancements that support either CPU and GPU processing. For CPU processing (formerly the only option), now you must manually load the [RegisterMismatch package](https://github.com/HolyLab/RegisterMismatch.jl): `using RegisterMismatch, RegisterQD`. For GPU processing, you should instead load the [RegisterMismatchCuda package](https://github.com/HolyLab/RegisterMismatchCuda.jl): `using RegisterMismatchCuda, RegisterQD`. *Note that loading both mismatch packages in the same session will cause method conflicts.* Both mismatch packages are registered in the publicly-available [HolyLabRegistry](https://github.com/HolyLab/HolyLabRegistry), and users are advised to add that registry. 

# version 0.2

## Breaking changes

- For users of `SD` ("space directions"), an input that allows you to specify the spatial sampling of your input array, the meaning of the returned transformation has changed. The return value is now in "physical space" rather than in "array index space." Specifically, formerly the returned transformation included the consequences of any change of scale needed to apply the transformation to the supplied arrays: for a rotation matrix `R`, the return was `SD\(R*SD)`.
Now just `R` gets returned. This means that a rigid transformation will now "look" rigid.
This change fixes some formerly-significant problems for
optimization when a known starting guess was available.
Use the new `arrayscale` function just before warping to prepare a transformation for a particular array with nonisotropic sampling.
