using CSV
using DataFrames
using Gadfly,Cairo,Fontconfig

# Load the CSV file
df = CSV.read("test.txt", DataFrame,delim=' ', footerskip=0,header=[:n, :Column2, :bottleneck]) 

println(df)

# Plot with Gadfly
plt=plot(df, x=:n, y=:bottleneck, Geom.point,Theme(background_color=colorant"white",default_color="red"), Geom.smooth(method=:lm))
draw(PNG("test.png", 6inch, 4inch),plt)
