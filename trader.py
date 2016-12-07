import time
import datetime

import poloniex
import observer
from autobahn.asyncio.wamp import ApplicationRunner


def setupPoloniex():
    file = open("james.champ.at.tuta.io.key", "r")
    key = file.readline()[0:-1]
    secret = file.readline()[0:-1]
    file.close()
    polo = poloniex.Poloniex(key, secret, timeout=10)
    return polo

def getCurrInfo(polo):
    allCurr = polo.api("returnCurrencies")
    return allCurr

#only call once
def saveChartData(polo, allCurrInfo):
    end = time.mktime(datetime.datetime.strptime("29.09.2016", "%d.%m.%Y").timetuple())
    start = time.mktime(datetime.datetime.strptime("29.09.2015", "%d.%m.%Y").timetuple())
    period = "300"

    for key in allCurrInfo.keys():
        currencyPair = "BTC_" + key
        result = polo.api("returnChartData", {"currencyPair": currencyPair, "period": period, "end": end, "start": start})
        print(currencyPair)
        if len(result) > 1:
            file = open("chart_data/" + currencyPair + ".csv", "w")
            for line in result:
                toWrite = str(line["quoteVolume"]) + ";" + str(line["weightedAverage"]) + ";" + str(line["open"]) + ";" + str(line["low"]) + ";" + str(line["volume"]) + ";" + str(line["date"]) + ";" + str(line["close"]) + ";" + str(line["high"]) + "\n"
                file.write(toWrite)
            file.close()
        else:
            print(result["error"])



polo = setupPoloniex()
allCurrInfo = getCurrInfo(polo)

#saveChartData(polo, allCurrInfo)



#runner = ApplicationRunner("wss://api.poloniex.com:443", "realm1")
#runner.run(observer.Observer)
