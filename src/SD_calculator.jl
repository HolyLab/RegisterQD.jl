"""
    getSD(A::AbstractArray)

If your image is not uniformily sampled, use this to get the `SD` matrix, which represents spacing along all axes of an image.
Please remember to strip out the time axis.

# Examples
```julia-repl
julia> myimage
Normed ImageMeta with:
  data: 3-dimensional AxisArray{N2f14,3,...} with axes:
    :x, 0.0 μm:0.71 μm:6.39 μm
    :l, 0.0 μm:0.71 μm:6.39 μm
    :z, 0.0 μm:6.2 μm:55.8 μm
And data, a 10×10×10 Array{N2f14,3} with eltype Normed{UInt16,14}
  properties:
    imagineheader: <suppressed>

julia> getSD(myimage)
3×3 SArray{Tuple{3,3},Float64,2,9} with indices SOneTo(3)×SOneTo(3):
 0.71  0.0   0.0
 0.0   0.71  0.0
 0.0   0.0   6.2
```
"""
function getSD(A::AbstractArray{T,N}) where {T,N} #only works on non-skewed arrays at the momoent.
    sd = spacedirections(A)
    SD = zeros(N,N)
    for i = 1:N
        for j = 1:N
            SD[i,j] = sd[i][j]/oneunit(sd[i][j])
        end
    end
    return SMatrix{N,N}(SD)
end
