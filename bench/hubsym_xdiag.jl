# Hubbard sector with XDiag 0.6 (Electron block with a permutation-group representation), eigvals_lanczos defaults.
# Usage: julia hubsym_xdiag.jl <spec file from hubsym_spec.py>
using XDiag
lines = readlines(ARGS[1])
h = split(lines[1]); L = parse(Int, h[1]); nup = parse(Int, h[2]); ndn = parse(Int, h[3]); U = parse(Float64, h[4])
bonds = Tuple{Int,Int}[]; perms = Vector{Int}[]; chars = Float64[]
for l in lines[2:end]
    w = split(l)
    if w[1] == "B"
        push!(bonds, (parse(Int, w[2]), parse(Int, w[3])))
    else
        push!(chars, parse(Float64, w[2])); push!(perms, parse.(Int, w[3:end]))
    end
end
t0 = time()
ops = OpSum()
for (i, j) in bonds
    global ops += "T" * Op("Hop", [i, j])
end
ops += "U" * Op("HubbardU")
ops["T"] = 1.0; ops["U"] = U
irrep = Representation(PermutationGroup([Permutation(p) for p in perms]), chars)
block = Electron(L, nup, ndn, irrep)
setup = time() - t0
t1 = time(); r = eigvals_lanczos(ops, block); t = time() - t1
println("{\"code\": \"xdiag\", \"spec\": \"$(basename(ARGS[1]))\", \"threads\": $(get(ENV, "OMP_NUM_THREADS", "?")), \"group_order\": $(length(perms)), \"dim\": $(Int(size(block))), \"setup_seconds\": $setup, \"iterations\": $(r.niterations), \"lanczos_seconds\": $t, \"seconds_per_iteration\": $(t / r.niterations), \"E0\": $(r.eigenvalues[1]), \"peak_rss_gib\": $(Sys.maxrss() / 2^30)}")
