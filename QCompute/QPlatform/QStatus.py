import traceback

import requests

from QCompute import Define
from QCompute.QPlatform import Error


def getIoPCASStatus():
    try:
        return requests.post(
            f"{Define.quantumHubAddr}/iopcas/status", json={
                "sess": Define.hubToken
            }).json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc())
