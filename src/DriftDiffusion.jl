module DriftDiffusion

using ForwardDiff

export adapt_clicks, compute_LL, DriftDiffusionHessian, DriftDiffusionVGH

function compute_LL(bup_times::Array{<:AbstractFloat},  bup_side::Array{<:Number},
    stim_dur::Array{<:AbstractFloat}, poke_r::Array{Bool}, input_params::Any ;
    nan_times=Array{Bool}(0), use_param=fill(true,1,9), param_default=[0 1 1 0 1 0.1 0 0.01 1],
    use_prior=zeros(1,9), prior_mu=zeros(1,9), prior_var=zeros(1,9), window_dt=0.01,
    adaptation_scales_perclick="var", nt=[])

    # validate inputs
    @assert size(bup_times,1)==size(bup_side,1) "bup_times and bup_side must have same 1st dimension length (i.e. number of maxbups)"
    @assert size(bup_side,2)==size(stim_dur,2) && size(bup_side,2)==size(poke_r,2) "bup_side, stim_dur, and poke_r must have the same 2nd dimension length (i.e. number of trials)"
    @assert length(use_param)==9 &&  length(use_param)==length(param_default) &&
        length(use_param)==length(use_prior) && length(use_param)==length(prior_mu) &&
        length(use_param)==length(prior_var) "use_param, param_default, use_prior, prior_mu, and prior_var must all have length 9 (because this is currently a 9-parameter model) ";
    if ~all(prior_var[use_prior.==0] .== 0)
        warn("some unused priors have nonzero variance")
    end
    if ~all(use_prior[prior_var.==0] .== 0)
        error("cannot use a prior with 0 variance")
    end

    # initialize variables
    # (Adrian: initializing some variables to be the same type as input_params is crucial, because
    # when you are using compute_LL to compute the Hessian, these must all be duals.)
    params_full              = zeros(eltype(input_params), 1,9)
    params_full[.!use_param] = param_default[.!use_param]
    params_full[use_param]   = input_params
    if params_full[end]<1 & isempty(nt)
        sz=size(bup_times);
        bup_times_normalized = broadcast(/,bup_times,stim_dur);
        nt = (1-params_full[end])/window_dt;
        bup_times = repmat(bup_times,1,nt);
        in_window = Array{Bool}(size(bup_times));
        for t=1:Int(round(nt))
            inds=(1:sz[2])+sz[2]*(t-1);
            time_window = [0 params_full[end]] + (t-1)*window_dt ;
            in_window[:,inds] = (bup_times_normalized.>time_window[1]) .& (bup_times_normalized.<=time_window[2]);
        end
        bup_times[.~in_window] = NaN;
        nan_times = isnan.(bup_times);
    elseif isempty(nt)
        nt=1;
    end
    nTrials = length(poke_r);
    prob_poked_r = zeros(eltype(input_params),nt,nTrials);
    NLL = zeros(eltype(input_params),1,nTrials);

    # calculate LL simultaneously for all trials, looping over time points
    for t=1:nt
        inds=(1:nTrials)+nTrials*(t-1);
        curr_buptimes = bup_times[:,inds];
       if isempty(nan_times)
           curr_nantimes = isnan.(curr_buptimes);
       else
           curr_nantimes = nan_times[:,inds];
       end
        # adapt those clicks
        if abs(params_full[5] - 1) > eps()
            adapted = adapt_clicks(curr_buptimes, curr_nantimes, params_full[5],params_full[6]) .* bup_side
        else
            adapted = copy(bup_side);
        end
        # apply integration timescale
        temp = stim_dur.-curr_buptimes
        temp = exp.(params_full[1]*temp)
        # compute mean of distribution
        temp2 = (adapted.*temp)
        mean_a = sum(temp2.*(.!isnan.(temp2)),1)
        # compute variance
        init_var    = max(eps(), params_full[4])
        a_var       = max(eps(), params_full[2])
        c_var       = max(eps(), params_full[3])
        # initial variance and accumulation variance
        if abs(params_full[1]) < 1e-10
            s2 = init_var*exp.(2*params_full[1]*stim_dur) + a_var*stim_dur
        else
            s2 = init_var*exp.(2*params_full[1]*stim_dur) + (a_var./(2*params_full[1]))*(exp.(2*params_full[1]*stim_dur)-1)
        end
        # add per click variance
        if adaptation_scales_perclick=="std"
            c2      = c_var .* abs.(adapted).^2 .* temp .^ 2
        elseif adaptation_scales_perclick=="var"
            c2      = c_var .* abs.(adapted) .* temp .^ 2
        elseif adaptation_scales_perclick=="none"
            c2      = c_var .* temp .^ 2
        end
        var_a   = s2 + sum(c2.*(.!isnan.(c2)),1)
        bias    = params_full[7]
        lapse   = min(max(params_full[8], eps()),  1-eps())
        erfTerm = erf.( -(bias-mean_a)./sqrt.(2*var_a));
        erfTerm[erfTerm.==1]=1-eps();
        erfTerm[erfTerm.==-1]=eps()-1;
        prob_poked_r[t,:]=((1-lapse).*(1+erfTerm)+lapse)/2 ;
    end
    prob_poked_r = mean(prob_poked_r,1);
    NLL[poke_r] = - ( log.( prob_poked_r[poke_r] ) );
    NLL[.!poke_r] = - ( log.(1- prob_poked_r[.!poke_r] ) );
    NLL_total = sum(NLL)
    # increment likelihood for priors
    for pp = find(use_prior)
        NLL_total += -(params_full[pp]-prior_mu[pp])^2/(2*prior_var[pp])
    end
    return NLL_total
end


function adapt_clicks(bup_times, nan_times, phi,tau_phi) #phi, tau_phi)
    adapt   = zeros(eltype(phi),size(bup_times)); # this must be the same type as the params because these could be duals
    adapt[.!nan_times] = 1
    phi     = max(0, phi)
    tau_phi = max(0, tau_phi)
    ici     = diff(bup_times)
    for i = 2:size(bup_times,1)
        adapt[i, :] = 1 + exp.(-ici[i-1,:]./tau_phi).*(adapt[i-1,:]*phi-1)
        adapt[i, ici[i-1, :] .<= 0] = 0
        adapt[i-1, ici[i-1,:] .<= 0] = 0
    end
    return adapt
end

function VGHfun(bup_times,  bup_side, stim_dur, poke_r, input_params; nan_times=fill(Bool,0), use_param=fill(true,1,9),
    param_default=[0 1 1 0 1 0.1 0 0.01 1],use_prior=zeros(1,9), prior_mu=zeros(1,9), prior_var=zeros(1,9),
    window_dt=0.01, adaptation_scales_perclick="var", nt=[])
    # compute hessian using autodiff
    # (Adrian: For reasons I don't understand, you need to call compute_LL with explicit keyword declaration for ForwardDiff to run
    optimFun(x) = DriftDiffusion.compute_LL(bup_times, bup_side, stim_dur, poke_r, x;nan_times=nan_times, use_param=use_param,
        param_default=param_default,use_prior=use_prior, prior_mu=prior_mu, prior_var=prior_var,
        window_dt=window_dt, adaptation_scales_perclick=adaptation_scales_perclick, nt=nt);
    out = DiffResults.HessianResult(zeros(size(input_params)));
    out = ForwardDiff.hessian!(out,optimFun,input_params);
    return DiffResults.value(out),DiffResults.gradient(out),DiffResults.hessian(out)
end

end # module
