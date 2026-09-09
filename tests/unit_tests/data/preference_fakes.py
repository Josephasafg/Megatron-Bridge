"""Chat-template stand-ins shared by the preference dataset and DPO builder tests."""

USER_HEADER, ASSISTANT_HEADER = 1, 2


class FakeChatTokenizer:
    """Header token per message and one token per word; ``add_generation_prompt`` appends the assistant header."""

    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            ids.append(ASSISTANT_HEADER if message["role"] == "assistant" else USER_HEADER)
            ids.extend(100 + len(word) for word in message["content"].split())
        if add_generation_prompt:
            ids.append(ASSISTANT_HEADER)
        return ids
