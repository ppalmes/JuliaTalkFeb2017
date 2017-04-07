#nprocs() == 1 && addprocs()

using GZip
using MNIST
using Knet
@everywhere using Compat

@everywhere logis(x) = 1 / (1 + exp(-x))
@everywhere dlogis(x) = logis(x) * (1 - logis(x))

#@everywhere logis(x) = max(0,x)
#@everywhere dlogis(x) = x>0?1:0
@everywhere rmse(x) = sqrt(mean(x.^2))

@everywhere function initParams(x,y,n_hidden,max_iter)
    global w1 = randn(n_hidden,size(x,1))
    global w2 = randn(n_hidden,n_hidden)
    global w3 = randn(size(y,1),n_hidden)  
    B1 = rand(n_hidden,size(y,1))
    B2 = rand(n_hidden,size(y,1))      
    errors = zeros(max_iter)
    outputs = zeros(size(y,2),max_iter)
    return (w1,w2,w3,errors,outputs,B1,B2)
end

@everywhere function ffward(x,y,w1,w2,w3)
    a1 = w1 * x
    h1 = logis.(a1)
    a2 = w2 * h1
    h2 = logis.(a2)
    ay = w3 * h2
    yhat = logis.(ay)
    err = yhat - y
    return(a1,h1,a2,h2,ay,yhat,err)
end

@everywhere function updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err;lr=0.1)
    w1 .-= δa1 * x' .* lr
    w2 .-= δa2 * h1' .* lr
    w3 .-= err * h2' .* lr
end

@everywhere function bp(dta,n_hidden,max_iter)
    (x,y)=dta[1]
    (w1,w2,w3,errors,outputs,_,_)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter   
        toterr=0
        for (x,y) in dta
            (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
            δa2 = (w3' * err) .* dlogis.(a2)
            δa1 = (w2' * δa2) .* dlogis.(a1)
            updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
            toterr += rmse(err)
        end
        println(toterr)
        errors[i] = toterr/length(dta)
    end
    return errors
end

@everywhere function dfa(dta,n_hidden,max_iter)
    (x,y)=dta[1]
    (w1,w2,w3,errors,outputs,B1,B2)=initParams(x,y,n_hidden,max_iter)
    for i = 1:max_iter    
        toterr=0
        for (x,y) in dta
            (a1,h1,a2,h2,ay,yhat,err) = ffward(x,y,w1,w2,w3)
            δa2 = (B2*err) .* dlogis.(a2) 
            δa1 = (B1*err) .* dlogis.(a1) 
            updateW!(w1,w2,w3,x,h1,h2,δa1,δa2,err)
            toterr += rmse(err)
        end
        println(toterr)
        errors[i] = toterr/length(dta)
    end
    return errors
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

@everywhere function loaddata()
    global xtrn,ytrn,xtst,ytst
    info("Loading MNIST...")
    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end
                        
@everywhere function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end


@everywhere function minibatch(x, y, batchsize; atype=Array{Float32}, xrows=784, yrows=10, xscale=255)
    xbatch(a)=convert(atype, reshape(a./xscale, xrows, div(length(a),xrows)))
    ybatch(a)=(a[a.==0]=10; convert(atype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
    xcols = div(length(x),xrows)
    xcols == length(y) || throw(DimensionMismatch())
    data = Any[]
    for i=1:batchsize:xcols-batchsize+1
        j=i+batchsize-1
        push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
    end
    return data
end

loaddata()
global dtrn = minibatch(xtrn, ytrn, 100; atype=Array{Float32})
global dtst = minibatch(xtst, ytst, 100; atype=Array{Float32})


#x = [1 1 ; 0 1 ; 1 0 ; 0 0]'
#y = [0 1 1 0]

n_hidden = 50; # Number of hidden units
max_iter = 10; #Number of learning steps
trials = 1;

#dfa_err=@parallel (hcat) for _=1:trials
#    dfa_errors = dfa(dtrn,n_hidden,max_iter)
#    dfa_errors
#end

bp_err = @parallel (hcat) for _=1:trials
    bp_errors = bp(dtrn,n_hidden,max_iter)
    bp_errors
end

#(bp_yhat,_) = bp(x,y,n_hidden,max_iter)
#
#ifa_err=@parallel (hcat) for _=1:trials
#    (_,ifa_errors) = ifa(x,y,n_hidden,max_iter)
#    ifa_errors
#end
#(ifa_yhat,_) = ifa(x,y,n_hidden,max_iter);
#fa_err=@parallel (hcat) for _=1:trials
#    (_,fa_errors) = fa(x,y,n_hidden,max_iter)
#    fa_errors
#end
#(fa_yhat,_) = fa(x,y,n_hidden,max_iter);
#using Plots
#l = @layout [
#    a b
#    c d
#    e f
#]

#unicodeplots()
##init_notebook(true)
#p1=plot(dfa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="DFA")
#p2=plot(ifa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="IFA")
#p3=plot(fa_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="FA")
#p4=plot(bp_yhat',label=["1,1" "0,1" "1,0" "0,0"],title="BP")
#p5=plot([dfa_err[:,1] ifa_err[:,1] fa_err[:,1] bp_err[:,1]],label=["DFA" "IFA" "FA" "BP"],title="First Trial Error")
#p6=plot([mean(dfa_err,2) mean(ifa_err,2) mean(fa_err,2) mean(bp_err,2)],label=["DFA" "IFA" "FA" "BP"],title="Mean Trials Error")
#plot(p1,p2,p3,p4,p5,p6,layout=l)
##plot(p1,p2,p3,p4,p5,p6,size=(900,900),layout=l)
