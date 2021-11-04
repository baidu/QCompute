import traceback

import requests

from QCompute import Define
from QCompute.QPlatform import Error, ModuleErrorCode

FileErrorCode = 8


def getIoPCASStatus():
    try:
        return requests.post(
            f"{Define.quantumHubAddr}/iopcas/status").json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 1)
