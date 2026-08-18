class MemoryService:

    def __init__(self):
        self.messages = []
    def add_message(
        self,
        role,
        content
    ):
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )
    def get_messages(self):
        return self.messages
    def get_recent_messages(
        self,
        limit=10
    ):
        return self.messages[-limit:]
    def clear(self):
        self.messages.clear()
