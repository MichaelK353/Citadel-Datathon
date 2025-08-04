require(rugarch)

pakData <- read.csv('test_packages.csv')
str(pakData)
pakData <- pakData[c('test_week', 'clickrate')]
pakData$test_week <- as.Date(pakData$test_week)
modData <- aggregate(pakData$clickrate, by=list(Category=pakData$test_week), FUN=mean)
# remove integrated component
diffPakData <- diff(modData$x)
# fitting ARMA(0,1)+GARCH(1,1) 
spec <- ugarchspec(variance.model = list(model = "sGARCH", 
                   garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 1)))
garch <- ugarchfit(spec = spec, data = diffPakData, solver.control = list(trace=0))
garch@fit$coef
garch@fit$matcoef
