#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from utils import open_redis_conn
from .Reader import Reader


class RedisReader(Reader):
    DATA_TYPES = ("txt", "img",)

    def __init__(self, data_type: str = "txt", redis_client=open_redis_conn()):
        super().__init__(filePath="")
        if data_type not in self.DATA_TYPES:
            raise TypeError(f"Argument data_type must be one of these types {self.DATA_TYPES}")
        self.redis_client = redis_client
        self.data_type = data_type

    def read(self) -> np.ndarray:
        keys = self.redis_client.keys(f"{self.data_type}-*")
        embeddings = []
        for key in keys:
            raw = self.redis_client.hgetall(key)[b"embedding"]
            embedding = np.frombuffer(raw, dtype=np.float32)
            embeddings.append(embedding)
        data = np.ndarray(shape=(1000, 768), buffer=np.array(embeddings), dtype=np.float32)
        return data


if __name__ == '__main__':
    rr = RedisReader()
    rr.read()
