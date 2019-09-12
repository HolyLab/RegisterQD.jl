# version 0.2

## Breaking changes

- For users of `SD` ("space directions"), an input that allows you to specify the spatial sampling of your input array, the meaning of the returned transformation has changed. The return value is now in "physical space" rather than in "array index space." Specifically, formerly the returned transformation included the consequences of any change of scale needed to apply the transformation to the supplied arrays: for a rotation matrix `R`, the return was `SD\(R*SD)`.
Now just `R` gets returned. This means that a rigid transformation will now "look" rigid.
This change fixes some formerly-significant problems for
optimization when a known starting guess was available.
Use the new `arrayscale` function just before warping to prepare a transformation for a particular array with nonisotropic sampling.
