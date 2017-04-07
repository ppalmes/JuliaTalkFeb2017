nprocs()==1 && addprocs()

@everywhere logis(x) = 1 ./ (1 .+ exp(-x))
@everywhere dlogis(x) = logis(x) .* (1 .- logis(x))

@everywhere function initParams(x,y,n_hidden,max_iter)
    w1 = randn(n_hidden,size(x,1))
    w2 = randn(n_hidden,n_hidden)
    w3 = randn(size(y,1),n_hidden)  
    B1 = rand(n_hidden,1)
    B2 = rand(n_hidden,1)      
    errors = zeros(max_iter)
    outputs = zeros(size(y,2),max_iter)
    return (w1,w2,w3,errors,outputs,B1,B2)
end

@everywhere function ffward(x,y,w1,w2,w3)
    a1 = w1 * x
    h1 = logis(a1)
    a2 = w2 * h1
    h2 = logis(a2)
    ay = w3 * h2
    yhat = logis(ay)
    err = yhat - y
    return(a1,h1,a2,h2,ay,yhat,err)
end

@everywhere function updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err;lr=1)
    w1 .-= δa1 * x' .* lr
    w2 .-= δa2 * h1' .* lr
    w3 .-= err * h2' .* lr
end

@everywhere function bp(x,y,n_hidden,max_iter)
    (w1,w2,w3,errors,outputs,_,_)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter   
        (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
        δa2 = (w3' * err) .* dlogis(a2)
        δa1 = (w2' * δa2) .* dlogis(a1)
        updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
        errors[i] = sum(abs(err))
        outputs[:,i] = yhat
    end
    return (outputs,errors)
end

@everywhere function fa(x,y,n_hidden,max_iter)
    (w1,w2,w3,errors,outputs,B1,B2)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter    
        (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
        δa2 = (B2.*err) .* dlogis(a2) 
        δa1 = (B1.*δa2) .* dlogis(a1) 
        updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
        errors[i] = sum(abs(err))
        outputs[:,i] = yhat
    end
    return (outputs,errors)
end

@everywhere function dfa(x,y,n_hidden,max_iter)
    (w1,w2,w3,errors,outputs,B1,B2)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter    
        (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
        δa2 = (B2.*err) .* dlogis(a2) 
        δa1 = (B1.*err) .* dlogis(a1) 
        updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
        errors[i] = sum(abs(err))
        outputs[:,i] = yhat
    end
    return (outputs,errors)
end

@everywhere function ifa(x,y,n_hidden,max_iter)
    (w1,w2,w3,errors,outputs,B1,_)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter    
        (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
        δa1 = (B1.*err) .* dlogis(a1) 
        δa2 = (w2 * δa1) .* dlogis(a2) 
        updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
        errors[i] = sum(abs(err))
        outputs[:,i] = yhat
    end
    return (outputs,errors)
end

# Example: XOR problem

x = [1 1 ; 0 1 ; 1 0 ; 0 0]'
y = [0 1 1 0]
n_hidden = 15; # Number of hidden units
max_iter = 500; #Number of learning steps
trials = 30;

dfa_err=@parallel (hcat) for _=1:trials
    (_,dfa_errors) = dfa(x,y,n_hidden,max_iter)
    dfa_errors
end
(dfa_yhat,_) = dfa(x,y,n_hidden,max_iter)

bp_err = @parallel (hcat) for _=1:trials
    (_,bp_errors) = bp(x,y,n_hidden,max_iter)
    bp_errors
end
(bp_yhat,_) = bp(x,y,n_hidden,max_iter)

ifa_err=@parallel (hcat) for _=1:trials
    (_,ifa_errors) = ifa(x,y,n_hidden,max_iter)
    ifa_errors
end
(ifa_yhat,_) = ifa(x,y,n_hidden,max_iter);

fa_err=@parallel (hcat) for _=1:trials
    (_,fa_errors) = fa(x,y,n_hidden,max_iter)
    fa_errors
end
(fa_yhat,_) = fa(x,y,n_hidden,max_iter);

using Plots
l = @layout [
    a b
    c d
    e f
]
gr()
p1=plot(dfa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="DFA")
p2=plot(ifa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="IFA")
p3=plot(fa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="FA")
p4=plot(bp_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="BP")
p5=plot([dfa_err[:,1] ifa_err[:,1] fa_err[:,1] bp_err[:,1]],label=["DFA" "IFA" "FA" "BP"],title="First Trial Error")
p6=plot([mean(dfa_err,2) mean(ifa_err,2) mean(fa_err,2) mean(bp_err,2)],label=["DFA" "IFA" "FA" "BP"],title="Mean Trials Error")
plot(p1,p2,p3,p4,p5,p6,size=(900,900),layout=l)
