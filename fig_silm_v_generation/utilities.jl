
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


function bV2I(bVector::Vector{Int})
    binaryVectorToInteger(bVector)
end

function binaryVectorToInteger(binVector::Vector{Int})
    if all(x -> x == 0 || x == 1, binVector)
        int_value = 0
        n = length(binVector)
        
        for (index, bit) in enumerate(reverse(binVector))
            int_value += bit * 2^(index - 1)
        end
        
        return int_value
    else
        throw(ArgumentError("Input vector must contain only zeros and ones"))
    end
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


    
    
