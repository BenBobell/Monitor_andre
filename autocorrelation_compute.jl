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


function monitoring_evolution_autocorrelation(L, gamma, dt, tmax, time_exponent, t_length)
    # U is Isometry that gives state. Initialized to Neel state. 
    U = zeros(L,trunc(Int, L/2))
    indices = CartesianIndex.(1:2:L, 1:trunc(Int, L/2))
    U[indices] .=1
    Corr_t = zeros(t_length)
    #U_tl_dagger: U(t+tau)^dagger
    U_tl_dagger = zeros(trunc(Int, L/2),L)
    for t in 1:tmax
        # etas gives the Ito steps in the Wiener process
        etas = rand(Distributions.Normal(0, sqrt(dt*gamma)), L)
        # D is the correlation matrix
        D = U*U'
        # Provides averaged C(t) starting at tmax-t_length
        if t == tmax-t_length+1
          U_tl_dagger = U'
        end
        if t>= tmax-t_length+1 
            D_t = U*U_tl_dagger 
            Corr_t[t-(tmax-t_length)] = Statistics.mean((abs.(LinearAlgebra.diag(D_t))) .^ 2)
        end
        # Gives array of <n_i>
        n_expected = LinearAlgebra.diag(D)
        # Provides measurement matrix
        M_diag = etas + (gamma*dt) * (2 * n_expected .- 1)
        M = LinearAlgebra.Diagonal(M_diag)
        #Time evolution for U(t+dt)
        U_dt = matrix_exponent_real(M) * time_exponent * U
        #QR decomposition that normalizes U_dt.
        Q, = LinearAlgebra.qr(U_dt)
        U = Matrix(Q)
    
    end
    
    return Corr_t

end

function run_iterations(L, gamma, W, t, dt, tmax, t_length, iterations)
    #Returns spatial correlation function C(r) for long times.
    #Note that sum over trajectories isn't averaged yet
    Corr_t_array  = zeros(t_length)
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
        Corr = monitoring_evolution_autocorrelation(L, gamma, dt, tmax, time_exponent,t_length)
        Corr_t_array += Corr
        end
    return Corr_t_array
end


L = 256
gamma = .02
W = 1.5
t = 1
dt = .05 
t_length = trunc(Int,1000/dt)
tmax = trunc(Int, 2000/dt)
iterations = 5 
#If parallelizing, idx will provide indexing for each set of trajectories
idx = 0

Corr_t_arr = run_iterations(L, gamma, W, t, dt, tmax, t_length, iterations)
JLD.save("Ct_data_$(gamma)_$(W)_$(idx).jld", "Corr_t", Corr_t_arr)

