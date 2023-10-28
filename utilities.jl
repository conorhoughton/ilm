
using Random

function makeRandom(n::Int)
    rand(0:1,n)
end

function makeXY(n::Int,sampleN::Int)

    x=Vector{Int64}[]
    y=Vector{Int64}[]

    for _ in 1:sampleN
        push!(x,makeRandom(n))
        push!(y,makeRandom(n))
    end

    (x,y)

end

function v2BV(bitN::Int,value::Int)
    valueToBinaryVector(bitN,value)
end

function valueToBinaryVector(bitN::Int,value::Int)
    binary_vector = Vector{Int}(undef, bitN)
    
    # Perform bitwise operations to convert v to binary
    for i in 1:bitN
        binary_vector[i] = value & 1
        value >>= 1
    end
    
    reverse(binary_vector)
    
end

function probabilityOfM(message,prob)
    p=1.0
    for i in 1:length(prob)
        p*=1.0+2.0*message[i]*prob[i]-message[i]-prob[i]
    end
    p
end

function calculateEntropy(p::Float64)
    
    if p==0 || p==1.0
        return 0.0
    end

    -p*log2(p)-(1-p)*log2(1-p)

end


    
    
