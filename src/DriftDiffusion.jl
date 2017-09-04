module DriftDiffusion

using MAT, ForwardDiff

export make_adapted_cat_clicks, compute_LL, compute_trial


function compute_LL(data, params)


# Set up variables
NLL = 0;
if length(params) == 8
    bias  = params[7];
    lapse = params[8];
elseif length(params) == 7
#    bias  = params[6];
#    lapse = params[7];
    bias = params[7];
    lapse = 0;
else
    bias  = params[5];
    lapse = params[6];
end

# iterate over trials
for i=1:length(data["pokedR"])
    ma,va = compute_trial(data,i,params);

    # compute pr, pl with bias
    pr = 0.5*(1+erf( -(bias-ma)/sqrt(2*va)));
    pl = 1-pr;

    # compute pr, pl with lapse
    PR = (1-lapse)*pr + lapse*0.5;
    PL = (1-lapse)*pl + lapse*0.5;


    # checking for log() stability
#    if PR == 0
#        PR = eps();
#    end
#    if PL == 0
#        PL = eps();
#    end

    # compute NLL for this trial
    if convert(Bool,data["pokedR"][i])
        nll = -log(PR);
    else
        nll = -log(PL);
    end

    # add to total over all trials
    NLL += nll;
end

#on brodycomp: sum(p.prior.*params)
# prior: 0.31, 0.94
#NLL += params[2]*0.31 + params[4]*0.94;
return NLL
end



function compute_trial(data, i, params);

    # run clicks through the adaptation process
    if length(params) == 8
        cl, cr = make_adapted_cat_clicks(data["leftbups"][i], data["rightbups"][i], params[5],params[6]);
    elseif length(params) == 7
#        cl, cr = make_adapted_cat_clicks(data["leftbups"][i], data["rightbups"][i], params[4],params[5]);
        cl, cr = make_adapted_cat_clicks(data["leftbups"][i], data["rightbups"][i], params[5],params[6]);
    elseif length(params) == 6;
        cl, cr = make_adapted_cat_clicks(data["leftbups"][i], data["rightbups"][i], params[3],params[4]);
    end

if isempty(cl)
  clicks=cr;
  times =  data["rightbups"][i];

elseif isempty(cr)
  clicks=-cl;
  times = data["leftbups"][i];

else
    clicks = [-cl cr];
    times = [data["leftbups"][i] data["rightbups"][i]];

end

    # compute mean of distribution
    mean_a = 0;
    for j=1:length(clicks)
        mean_a += clicks[j]*exp(params[1]*(data["T"][i]-times[j]));
    end

    # compute variance of distribution
    # three sources: initial (params[4]), accumulation (params[2]), and per-click (params[3])

    if length(params) == 8
        a_var    = params[2];
        c_var    = params[3];
        init_var = params[4];
    elseif length(params) == 7
#        a_var    = params[2];
#        c_var    = params[3];
#        init_var = 0;
        a_var    = params[2];
        c_var    = params[3];
        init_var = params[4];
    elseif length(params) == 6
        a_var    = 0;;
        c_var    = params[2];
        init_var = 0;
    end

    # Initial and accumulation variance
    if abs(params[1]) < 1e-10
        s2 = init_var*exp(2*params[1]*data["T"][i]) + a_var*data["T"][i];
    else
        s2 = init_var*exp(2*params[1]*data["T"][i]) + (a_var/(2*params[1]))*(exp(2*params[1]*data["T"][i])-1);
    end

    # add per-click variance
    for j=1:length(clicks)
        s2 += c_var*abs(clicks[j])*exp(2*params[1]*(data["T"][i] - times[j]));
    end

    var_a = s2;

    # return mean and variance of distribution
    return mean_a, var_a
end


# Adaptation function with both within stream and across stream adaptation
function make_adapted_cat_clicks(leftbups, rightbups, phi, tau_phi)
if isempty(leftbups)
  L = leftbups;
  R = rightbups;
  return L,R
end
if isempty(rightbups)
  L = leftbups;
  R = rightbups;
  return L,R
end
    if abs(phi - 1) > eps()
        lefts  = [leftbups;  -ones(1,length(leftbups))];
        rights = [rightbups; +ones(1,length(rightbups))];
        allbups = sortrows([lefts rights]')'; # one bup in each col, second row has side bup was on

        if length(allbups) <= 1
            ici = [];
        else
            ici = (allbups[1,2:end]  - allbups[1,1:end-1])';
        end

        adapted = ones(typeof(phi), 1, size(allbups,2));

        for i = 2:size(allbups,2)
            if ici[i-1] <= 0
            adapted[i-1] = 0;
            adapted[i] =0;
            else
#            last = tau_phi * log(1 - adapted[i-1]*phi);
#            adapted[i] = 1 - exp((-ici[i-1] + last)/tau_phi);
            adapted[i] = 1+ exp(-ici[i-1]/tau_phi)*(adapted[i-1]*phi -1);
            end
        end

    	adapted = real(adapted);

    	L = adapted[allbups[2,:] .==-1]';
    	R = adapted[allbups[2,:] .==+1]';
    else
    	# phi was equal to 1, there's no adaptation going on.
    	L = leftbups;
    	R = rightbups;
    end

    return L, R
end




# package code goes here

end # module
