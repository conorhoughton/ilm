using CSV
using DataFrames,GLM
using Gadfly,Cairo,Fontconfig

# Load the CSV file
df = CSV.read("fig3.csv", DataFrame,delim=',', footerskip=0)
#,header=[:n, :generation, :bottleneck]) 

println(df)

# Plot with Gadfly


plt=plot(df, x=:n, y=:bottleneck, Geom.point,Theme(background_color=colorant"white",default_color="red"), Geom.smooth(method=:lm))
draw(PNG("fig3_nVbottle.png", 5inch, 3inch),plt)

model = lm(@formula(bottleneck ~ n), df)

coefficients = coef(model)
println("Intercept: ", coefficients[1], " Slope: ", coefficients[2])


plt=plot(layer(df, x=:n, y=:generations, Geom.point),
         layer(df, x=:n, y=:generations, Geom.line),
         Theme(background_color=colorant"white",default_color="red")
         )
draw(PNG("fig3_nVgen.png", 2.5inch, 3inch),plt)
