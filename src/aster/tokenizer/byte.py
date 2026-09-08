def encode(text):
    data = text.encode("utf-8")
    return list(data)


def decode(ids):
    data = bytes(ids)
    return data.decode("utf-8")