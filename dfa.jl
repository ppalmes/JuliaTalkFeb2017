nprocs()==1 && addprocs()

@everywhere logis(x) = 1 ./ (1 .+ exp(-x))
@everywhere dlogis(x) = logis(x) .* (1 .- logis(x))

@everywhere function dfa!(w1,w2,w3,B1,B2,inp,out)
    a1 = w1 * inp
    z1 = logis(a1)
    a2 = w2 * z1
    z2 = logis(a2)
    ay = w3 * z2
    y = logis(ay)
    err = y - out
    d_a1 = (B1.*err) .* dlogis(a1) 
    d_a2 = (B2.*err) .* dlogis(a2) 
    w1 .-= d_a1 * inp'
    w2 .-= d_a2 * z1'
    w3 .-= err * z2'
    return Dict("output"=>y,"err"=>err)
end

@everywhere function bp!(bp_w1,bp_w2,bp_w3,inp,out)
    bp_a1 = bp_w1*inp
    bp_z1 = logis(bp_a1)
    bp_a2 = bp_w2 * bp_z1
    bp_z2 = logis(bp_a2)
    bp_ay = bp_w3 * bp_z2
    bp_y = logis(bp_ay)
    bp_err = bp_y - out
    bp_d_a3 = (bp_err) .* dlogis(bp_ay)
    bp_d_a2 = (bp_w3' * bp_d_a3) .* dlogis(bp_a2)
    bp_d_a1 = (bp_w2' * bp_d_a2) .* dlogis(bp_a1)
    bp_w1 .-= bp_d_a1 * inp'
    bp_w2 .-= bp_d_a2 * bp_z1'
    bp_w3 .-= bp_d_a3 * bp_z2'
    return Dict("output"=>bp_y,"err"=>bp_err)
end

inp = [1 1 ; 0 1 ; 1 0 ; 0 0]'
out = [0 1 1 0]
y=zeros(size(out))
n_hidden = 15; # Number of hidden units
num_iterations = 1000; #Number of learning steps
trials = 50;

e_store = SharedArray(Float64,(num_iterations,trials))
bp_e_store = SharedArray(Float64,(num_iterations,trials))
y_store = SharedArray(Float64,(size(out,2),num_iterations))
bp_y_store = SharedArray(Float64,(size(out,2),num_iterations))

@elapsed @sync @parallel for jj = 1:trials
    w1 = randn(n_hidden,size(inp,1))
    w2 = randn(n_hidden,n_hidden)
    w3 = randn(size(out,1),n_hidden)
          
    B1=rand(n_hidden,1)
    B2=rand(n_hidden,1)    
    
    bp_w1 = randn(n_hidden,size(inp,1))
    bp_w2 = randn(n_hidden,n_hidden)
    bp_w3 = randn(size(out,1),n_hidden) 
    
    for ii = 1:num_iterations       
        dfa_res = dfa!(w1,w2,w3,B1,B2,inp,out)       
        bp_res = bp!(bp_w1,bp_w2,bp_w3,inp,out)       
        if jj == 1
            y_store[:,ii] = dfa_res["output"]'
            bp_y_store[:,ii] = bp_res["output"]'
        end       
        e_store[ii,jj] = sum(abs(dfa_res["err"]))
        bp_e_store[ii,jj] = sum(abs(bp_res["err"]))
    end
end

using Plots
l = @layout [
    a b
    c d
]

gr()
p1=plot(y_store',label=["1,1" "0,1" "1,0" "0,0"],title="DFA")
p2=plot(bp_y_store',label=["1,1" "0,1" "1,0" "0,0"],title="BP")
p3=plot([e_store[:,1] bp_e_store[:,1]],label=["DFA" "BP"],title="First Trial")
p4=plot([mean(e_store,2) mean(bp_e_store,2)],label=["DFA" "BP"],title="Mean Error")
plot(p1,p2,p3,p4,layout=l)
