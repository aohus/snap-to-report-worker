from fastapi import Request
import asyncio

def get_lock(request: Request) -> asyncio.Lock:
    return request.app.state.clusterer_lock
