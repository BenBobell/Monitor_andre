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
        # Gives array of <n_i>
        n_expected = sum(abs.(U) .^2, dims =(2))
        if t >= t_start
            U_4 = (abs.(U)) .^ 4
            IPR_avg += sum(U_4)
        end
 
        # Provides measurement matrix
        M_diag = etas + (gamma*dt) * (2 * n_expected .- 1)
        # Time evolution for U(t+dt) and QR decomposition 
        Q, = LinearAlgebra.qr(exp.(M_diag) .* time_exponent * U)
        # Return thin QR decomposition
        U = Matrix(Q)
    
    end
    
    return 2*IPR_avg/(L*(tmax-t_start+1))

end

function run_iterations(L, W, gamma, g, dt, tmax, t_start, iterations)
    #Returns IPR for long times.
    IPR =0 
    for iteration in 1:iterations
        # Declare Hamiltonian
        # H = g*\sum c_{i+1}^{\dagger}c_i +h.c + h_i n_i
        w = -W .+ (2*W) .* rand(L)
        H = zeros(L, L)
        H[LinearAlgebra.diagind(H,1)] .= g
        H[LinearAlgebra.diagind(H,-1)] .= g
        H[LinearAlgebra.diagind(H)] .= w
        H[1,L] = g
        H[L,1] = g
        time_exponent = matrix_exponent_imaginary(H,dt)
        IPR += monitoring_evolution_IPR(L, gamma, dt, tmax, t_start, time_exponent)
        end
    return IPR/iterations
end


L = 256
W_vals = .5:.1:4
gamma_vals = .01:.01:.1
g_vals = 1
dt = .05 
tmax = trunc(Int, 2000/dt)
t_start = tmax-50
iterations = 5 
#If parallelizing, idx will provide indexing for each set of trajectories
@time begin
IPR_vals = @. run_iterations(L, W_vals', gamma_vals, g_vals, dt,tmax, t_start, iterations)
end
JLD.save("IPR_data.jld", "IPR", IPR_vals)

