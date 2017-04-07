#!/usr/local/bin/julia

nprocs()==1 && addprocs()

using RDatasets
@everywhere using RDatasets
using DecisionTree
@everywhere using DecisionTree

@everywhere function irisAcc(treesize,sp)
	iris = dataset("datasets", "iris")
	features = convert(Array, iris[:, 1:4]);
	labels = convert(Array, iris[:, 5]);
	#model = build_forest(labels, features, 2, 10, 0.5, 6);
	accuracy = nfoldCV_forest(labels, features, 2, treesize, 3, sp);
	mean(accuracy)
end

@elapsed begin
	s=0
	n=500
	for i=1:n
		s += irisAcc(10,0.3)
	end
	s/n
end

@elapsed begin
	n=500
	s=@parallel (+) for i=1:n
		irisAcc(10,0.3)
	end
	s/n
end

@everywhere function trials(tr,sp)
	@sync res=@parallel (vcat) for i=1:10
		irisAcc(tr,sp)
	end
	return res
end

@elapsed begin
	trTable = @parallel (vcat) for tr in [10,20,50,100]
		spTable=@parallel (vcat) for sp in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0]
			res=@parallel (vcat) for i=1:10
				irisAcc(tr,sp)
			end
			[tr sp mean(res) std(res) length(res)]
		end
	end
end

sorted = sortrows(trTable,by=x->x[3],rev=true);
sorted = DataFrame(sorted);
rename!(sorted,Dict(:x1=>:treeSize,:x2=>:SP,:x3=>:ACC,:x4=>:SD,:x5=>:Trials))
