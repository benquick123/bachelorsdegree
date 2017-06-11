
class Conversation:

    def __init__(self, conversation_id, currencies=[], date=0, messages=[]):
        self.conversation_id = conversation_id
        self.currencies = currencies
        self.date_start = date

        self.messages = messages
        self.sentiment = {}

    def __str__(self):
        return ""


class Message:

    def __init__(self, message_id, currencies=[], date=0, text="", author=""):
        self.message_id = message_id
        self.currencies = currencies
        self.date = date
        self.text = text
        self.stemmed_text = ""
        self.author = author

        self.sentiment_text = {}

    def __str__(self):
        return ""

