module DriftDiffusion

using MAT, ForwardDiff

export adapt_clicks, compute_LL, DriftDiffusionHessian





function compute_LL(bup_time,  bup_side, stim_dur, poke_r, input_params; nan_time=[], use_param=fill(true,8), param_default=[0, 1, 1, 0, 1, 0.1, 0, 0.01],use_prior=zeros(8), prior_mu=zeros(8), prior_var=zeros(8))

if isempty(nan_time)
    nan_time = fill(false, size(bup_time));
end
if ~all(prior_var[use_prior.==0] .== 0)
    warn("some unused priors have nonzero variance")
end
if ~all(use_prior[prior_var.==0] .== 0)
    error("cannot use a prior with 0 variance")
end


params              = zeros(eltype(input_params), 8)
params[.!use_param] = param_default[.!use_param]
params[use_param]   = input_params

# allocate array to store negative log likelihoods for each trial
NLL     = zeros(eltype(input_params), length(poke_r), 1)

# adapt those clicks
if abs(params[5] - 1) > eps()
    #adapted = adapt_clicks(bup_time, nan_time, params[5], params[6]) .* bup_side
    adapted = adapt_clicks(bup_time, nan_time, params) .* bup_side
else
    adapted = bup_time;
    adapted[.!nan_time] = 1
end

# apply integration timescale
temp = stim_dur.-bup_time
temp = exp.(params[1]*temp)

# compute mean of distribution
mean_a = (adapted.*temp)
mean_a = sum(mean_a.*(.!isnan.(mean_a)),1)

# compute variance
init_var    = max(eps(), params[4])
a_var       = max(eps(), params[2])
c_var       = max(eps(), params[3])

# initial variance and accumulation variance
if abs(params[1]) < 1e-10
    s2 = init_var*exp.(2*params[1]*stim_dur) + a_var*stim_dur
else
    s2 = init_var*exp.(2*params[1]*stim_dur) + (a_var./(2*params[1]))*(exp.(2*params[1]*stim_dur)-1)
end
# add per click variance
c2      = c_var .* abs.(adapted) .* temp .^ 2
var_a   = s2 + sum(c2.*(.!isnan.(c2)),1)

bias    = params[7]
lapse   = min(max(params[8], eps()),  1-eps())


pr = 0.5*(1+erf.( -(bias-mean_a)./sqrt.(2*var_a)))
pl = 1-pr

pl = (1-lapse)*pl+lapse/2
pr = (1-lapse)*pr+lapse/2
NLL = - log.(pr).*poke_r - log.(pl).*.!poke_r

NLL_total = sum(NLL)

# increment likelihood for priors
for pp = find(use_prior)
    NLL_total += -(params[pp]-prior_mu[pp])^2/(2*prior_var[pp])
end

return NLL_total
end


function adapt_clicks(bup_time, nan_time, params) #phi, tau_phi)
phi     = params[5]
tau_phi = params[6]
#adapt   = copy(bup_time)
adapt   = zeros(eltype(params), size(bup_time))
adapt[.!nan_time] = 1
phi     = max(0, phi)
tau_phi = max(0, tau_phi)
ici     = diff(bup_time)
xxx = 0;
for i = 2:size(bup_time,1)
    #prev = tau_phi * log.(1 - adapt[i-1,:]*phi)
    #adapt[i,:] = 1 - exp.( (-ici[i-1,:] + prev) / tau_phi)
    adapt[i, :] = 1 + exp.(-ici[i-1,:]./tau_phi).*(adapt[i-1,:]*phi-1)
    adapt[i, ici[i-1, :] .<= 0] = 0
    adapt[i-1, ici[i-1,:] .<= 0] = 0
end

#adapt = real(complex(adapt))
return adapt
end


function DriftDiffusionHessian(data,x,params)
    # compute hessian using autodiff
    autodiff_hessian = ForwardDiff.hessian(x->compute_LL(data,x), params)
    return autodiff_hessian
end

# package code goes here

end # module
