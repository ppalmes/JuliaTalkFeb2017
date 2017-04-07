using Knet,ArgParse,Compat,GZip
using MNIST

function drelu(x)
    return x>0?1:0
end

function predict!(x,w,ao,ho)

    j=1
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        ao[j].=x
        if i<length(w)-1
            x = relu(x) # max(0,x)
            ho[j].=x
        end
        j+=1
        println(j)
    end

    return x
end

function ffward!(x,ygold,w,ao,ho)
    ypred=predict!(x,w,ao,ho)
    ynorm = logp(ypred,1)
    err = -sum(ygold .* ynorm) / size(ygold,2)
    #err = ypred - ygold
    return err
end

function updateParams!(x,w,ho,δao,err;lr=0.5)
    w[1] -= ((δao[1] * x') .* lr)
    j=2
    for i=3:2:(length(w)-2)
        w[i] -= ((δao[j] * ho[j-1]') .* lr)
        j+=1
    end
    #w[length(w)-1] -= ((fill!(zeros(10,100),err) * ho[j-1]') .* lr)
end

function dfa_train(w,ao,ho,B, dtrn; lr=.5, epochs=10)

    δao = Any[]
    for epoch=1:epochs
        for (x,y) in dtrn
            err = ffward!(x, y,w,ao,ho)
            for i=1:(length(ao)-1)
                push!(δao,(B[i].*err) .* drelu.(ao[i]))
            end
            updateParams!(x,w,ho,δao,err;lr=lr)
            println(sum(w[3]))
        end
    end

    return w
end

function accuracy(w,ao,ho, dtst)

    ncorrect = ninstance = nloss = 0

    for (x, ygold) in dtst

        ypred = predict!(x,w,ao,ho)

        ynorm = ypred .- log(sum(exp(ypred),1))
        nloss += -sum(ygold .* ynorm)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold,2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

function initParams(h,batchsize; atype=Array{Float32}, winit=0.1)
    w = Any[]
    ao = Any[]
    ho = Any[]
    B = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(atype, winit*randn(y,x)))
        push!(w, convert(atype, randn(y, 1)))
        push!(ao,convert(atype,zeros(y,batchsize)))
        push!(ho,convert(atype,zeros(y,batchsize)))
        push!(B,convert(atype,rand(y,x)))
        x = y
    end
    return (w,ao,ho,B)
end

function loaddata()
    global xtrn,ytrn,xtst,ytst
    info("Loading MNIST...")
    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function minibatch(x, y, batchsize; atype=Array{Float32}, xrows=784, yrows=10, xscale=255)
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

#function main()

    o=Dict{Symbol,Any}()
    o[:seed]=-1; o[:batchsize]=100; o[:hidden]=[100;50;30;20];
    o[:epochs]=10; o[:lr]=0.5; o[:atype]="Array{Float32}";
    o[:gcheck]=0; o[:winit]=0.1; o[:fast]=false;

    if !o[:fast]
        println("opts=",[(k,v) for (k,v) in o]...)
    end

    atype = eval(parse(o[:atype]))

    (w,ao,ho,B) = initParams(o[:hidden],o[:batchsize]; atype=atype, winit=o[:winit])

    if !isdefined(MNIST,:xtrn); loaddata(); end

    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; atype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; atype=atype)

    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,ao,ho,dtrn),:tst,accuracy(w,ao,ho,dtst)))

    @time for epoch=1:o[:epochs]
        dfa_train(w,ao,ho,B,dtrn; lr=o[:lr], epochs=1)
        report(epoch)
    end

    return w
#end

#main()
