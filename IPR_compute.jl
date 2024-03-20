import LinearAlgebra
import Random
import Distributions
import Statistics
import JLD

# complex time evolution. Matrix exponentiaton performed w/ diagonalization 
# M is the matrix to time evolve, dt is the time step
function matrix_exponent_imaginary(M, dt)
    eigenvals, eigenvectors = LinearAlgebra.eigen(M)
    expD = exp.(-dt*im*eigenvals)
    return eigenvectors * LinearAlgebra.Diagonal(expD) * eigenvectors'
end

# Real matrix exponentiation performed w/ diagonalization.
function matrix_exponent_real(M)
    eigenvals, eigenvectors = LinearAlgebra.eigen(M)
    expD = exp.(eigenvals)
    return eigenvectors * LinearAlgebra.Diagonal(expD) * eigenvectors'
end

# compute Von neumann entropy of subsystem A from correlation matrix
# D is correlation matrix
# Astart, Aend gives start/end coordinates for subsystem.
function vnEntropy(D, Astart, Aend)
    D_Arestrict = D[Astart:Aend, Astart:Aend]
    eigenvals, = LinearAlgebra.eigen(D_Arestrict)
    
    #filter out 0log0 and negative terms
    eigen_filter = filter(x->x>0 && x<1,eigenvals)
    
    S_p = LinearAlgebra.dot(eigen_filter, log.(eigen_filter))
    S_q = LinearAlgebra.dot(1 .- eigen_filter, log.(1 .- eigen_filter))
    return -(S_p + S_q)
end


function monitoring_evolution_IPR(L, gamma, dt, tmax, t_start, time_exponent)
    # U is Isometry that gives state. Initialized to Neel state. 

    U = zeros(L,trunc(Int, L/2))
    indices = CartesianIndex.(1:2:L, 1:trunc(Int, L/2))
    U[indices] .=1
    IPR_avg = 0
    for t in 1:tmax
        # etas gives the Ito steps in the Wiener process
        etas = rand(Distributions.Normal(0, sqrt(dt*gamma)), L)
        # D is the correlation matrix
        D = U*U'
        if t >= t_start
            IPR_avg += sum(abs.(LinearAlgebra.diag(D)) .^ 2)
        end
        # Gives array of <n_i>
        n_expected = LinearAlgebra.diag(D)
        # Provides measurement matrix
        M_diag = etas + (gamma*dt) * (2 * n_expected .- 1)
        M = LinearAlgebra.Diagonal(M_diag)
        # Time evolution for U(t+dt)
        U_dt = matrix_exponent_real(M) * time_exponent * U
        # QR decomposition that normalizes U_dt
        Q, = LinearAlgebra.qr(U_dt)
        U = Matrix(Q)
    
    end
    
    return 2*IPR_avg/(L*(tmax-t_start))

end

function run_iterations(L, gamma, W, t, dt, tmax, t_start, iterations)
    #Returns IPR for long times.
    IPR =0 
    for iteration in 1:iterations
        # Declare Hamiltonian
        # H = t*\sum c_{i+1}^{\dagger}c_i +h.c + h_i n_i
        w = -W .+ (2*W) .* rand(L)
        H = zeros(L, L)
        H[LinearAlgebra.diagind(H,1)] .= t
        H[LinearAlgebra.diagind(H,-1)] .= t
        H[LinearAlgebra.diagind(H)] .= w
        H[1,L] = t
        H[L,1] = t
        time_exponent = matrix_exponent_imaginary(H,dt)
        IPR += monitoring_evolution_IPR(L, gamma, dt, tmax, t_start, time_exponent)
        end
    return IPR/iterations
end


L = 256
W_vals = .5:.1:4
gamma_vals = .01:.01:.1
t_vals = 1
dt = .05 
tmax = trunc(Int, 2000/dt)
t_start = tmax-50
iterations = 5 
#If parallelizing, idx will provide indexing for each set of trajectories
@time begin
IPR_vals = @. run_iterations(L, gamma_vals', W_vals, t_vals, dt,tmax, t_start, iterations)
end
JLD.save("IPR_data.jld", "Corr_r", Corr_r_arr)

