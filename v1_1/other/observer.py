from autobahn.asyncio.wamp import ApplicationSession

from asyncio import coroutine


class Observer(ApplicationSession):

    def onConnect(self):
        self.join(self.config.realm)

    @coroutine
    def onJoin(self, details):
        def onTicker(*args):
            print(*args)

        print(self.subscribe(onTicker, 'ticker'))


