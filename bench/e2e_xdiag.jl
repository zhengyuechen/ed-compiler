# End-to-end ground-state energy with XDiag (eigvals_lanczos, its default precision 1e-12, seed 42).
# Usage: julia e2e_xdiag.jl <model>      model = ring:L[:sym] | j1j2:L:J2[:sym] | square:LxxLy[:sym]
using XDiag
name = ARGS[1]; parts = split(name, ':'); kind = parts[1]
function bonds_and_group(parts)
    if kind == "ring" || kind == "j1j2"
        L = parse(Int, parts[2]); J2 = kind == "j1j2" ? parse(Float64, parts[3]) : 0.0
        b = [(i, mod1(i + 1, L), 1.0) for i in 1:L]
        J2 != 0 && append!(b, [(i, mod1(i + 2, L), J2) for i in 1:L])
        perms = [[mod1(i + k, L) for i in 1:L] for k in 0:L-1]
        rest = kind == "j1j2" ? parts[4:end] : parts[3:end]
        return L, b, perms, rest
    else
        Lx, Ly = parse.(Int, split(parts[2], 'x')); L = Lx * Ly
        site(x, y) = mod(x, Lx) + Lx * mod(y, Ly) + 1
        b = vcat([(site(x, y), site(x + 1, y), 1.0) for y in 0:Ly-1 for x in 0:Lx-1],
                 [(site(x, y), site(x, y + 1), 1.0) for y in 0:Ly-1 for x in 0:Lx-1])
        perms = [[site((i - 1) % Lx + dx, (i - 1) ÷ Lx + dy) for i in 1:L] for dy in 0:Ly-1 for dx in 0:Lx-1]
        return L, b, perms, parts[3:end]
    end
end
L, b, perms, rest = bonds_and_group(parts)
sym = !isempty(rest) && rest[1] == "sym"
t0 = time()
ops = OpSum()
for (i, j, J) in b
    global ops += J * Op("SdotS", [i, j])
end
block = sym ? Spinhalf(L, L ÷ 2, Representation(PermutationGroup([Permutation(p) for p in perms]), ones(length(perms)))) : Spinhalf(L, L ÷ 2)
setup = time() - t0
d = Int(size(block))
t1 = time()
r = eigvals_lanczos(ops, block)
t = time() - t1
println("{\"code\": \"xdiag\", \"model\": \"$name\", \"threads\": $(get(ENV, "OMP_NUM_THREADS", "?")), \"dim\": $d, \"setup_seconds\": $setup, \"iterations\": $(r.niterations), \"lanczos_seconds\": $t, \"seconds_per_iteration\": $(t / r.niterations), \"E0\": $(r.eigenvalues[1]), \"criterion\": \"$(r.criterion)\", \"peak_rss_gib\": $(Sys.maxrss() / 2^30)}")
