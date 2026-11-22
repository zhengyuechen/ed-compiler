# Hubbard ground-state energy with XDiag (Electron block, Hop + HubbardU), eigvals_lanczos defaults.
# Usage: julia e2e_hubbard_xdiag.jl chain:L:U:nup:ndn | square:LxxLy:U:nup:ndn
using XDiag
p = split(ARGS[1], ':')
U = parse(Float64, p[3]); nup = parse(Int, p[4]); ndn = parse(Int, p[5])
if p[1] == "chain"
    L = parse(Int, p[2]); b = [(i, mod1(i + 1, L)) for i in 1:L]
else
    Lx, Ly = parse.(Int, split(p[2], 'x')); L = Lx * Ly
    site(x, y) = mod(x, Lx) + Lx * mod(y, Ly) + 1
    b = vcat([(site(x, y), site(x + 1, y)) for y in 0:Ly-1 for x in 0:Lx-1], [(site(x, y), site(x, y + 1)) for y in 0:Ly-1 for x in 0:Lx-1])
end
t0 = time()
ops = OpSum()
for (i, j) in b
    global ops += "T" * Op("Hop", [i, j])
end
ops += "U" * Op("HubbardU")
ops["T"] = 1.0; ops["U"] = U
block = Electron(L, nup, ndn)
setup = time() - t0
t1 = time(); r = eigvals_lanczos(ops, block); t = time() - t1
println("{\"code\": \"xdiag\", \"model\": \"$(ARGS[1])\", \"threads\": $(get(ENV, "OMP_NUM_THREADS", "?")), \"dim\": $(Int(size(block))), \"setup_seconds\": $setup, \"iterations\": $(r.niterations), \"lanczos_seconds\": $t, \"seconds_per_iteration\": $(t / r.niterations), \"E0\": $(r.eigenvalues[1]), \"peak_rss_gib\": $(Sys.maxrss() / 2^30)}")
