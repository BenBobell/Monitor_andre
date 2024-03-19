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


function monitoring_evolution(L, gamma, dt, tmax, time_exponent)
    # U is Isometry that gives state. Initialized to Neel state. 
    U = zeros(L,trunc(Int, L/2))
    indices = CartesianIndex.(1:2:L, 1:trunc(Int, L/2))
    U[indices] .=1
    S_t = zeros(tmax)
    for t in 1:tmax
        # etas gives the Ito steps in the Wiener process
        etas = rand(Distributions.Normal(0, sqrt(dt*gamma)), L)
        # D is the correlation matrix
        D = U*U'
        # Von Neumann entropy of subsystem A= [1,L/2] at time t.
        S_t[t] = vnEntropy(D,1,trunc(Int, L/2))
        # Gives array of <n_i>
        n_expected = LinearAlgebra.diag(D)
        #Provides measurement matrix
        M_diag = etas + (gamma*dt) * (2 * n_expected .- 1)
        M = LinearAlgebra.Diagonal(M_diag)
        #Time evolution for U(t+dt)
        U_dt = matrix_exponent_real(M) * time_exponent * U
        #QR decomposition that normalizes U_dt
        Q, = LinearAlgebra.qr(U_dt)
        U = Matrix(Q)
    
    end
    
    return S_t

end

function run_iterations(L, gamma, W, t, dt, tmax, tstart, iterations)
# S_t_array provies S(t). Note that it is the sum over trajectories,
# but it isn't averaged yet (so we don't divide by "iterations")
# S_inf_array provides the average S_inf value for a given trajectory

S_t_array = zeros(tmax)
S_inf_array = zeros(iterations)

    for iteration in 1:iterations
        # Declare Hamiltonian
        # H = t*\sum c_{i+1}^{\dagger}c_i +h.c + h_i n_i
        w = -W .+ (2*W) * rand(L)
        H = zeros(L, L)
        H[LinearAlgebra.diagind(H,1)] .= t
        H[LinearAlgebra.diagind(H,-1)] .= t
        H[LinearAlgebra.diagind(H)] .= w
        H[1,L] = t
        H[L,1] = t
        time_exponent = matrix_exponent_imaginary(H,dt)
        S = monitoring_evolution(L, gamma, dt, tmax, time_exponent)
        S_t_array += S
        S_inf_array[iteration] = Statistics.mean(S[tstart:tmax])

    end

return S_t_array, S_inf_array
end

L = 256
gamma = .02
W = 1.5
t = 1
dt = .05 
tmax = trunc(Int, 600/dt)
tstart = trunc(Int, 300/dt)
iterations = 5 
#If parallelizing, idx will provide indexing for each set of trajectories
idx = 0

S_t_arr, S_inf_arr = run_iterations(L, gamma,W, dt, tmax, tstart, iterations)
JLD.save("S_data_$(gamma)_$(W)_$(idx).jld", "St", S_t_arr, "Sinf", S_inf_arr)

