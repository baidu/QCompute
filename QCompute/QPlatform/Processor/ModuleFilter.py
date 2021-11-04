from typing import List, Optional

from QCompute import UnrollCircuitModule, CompressGateModule

from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform import BackendName, Error, ModuleErrorCode

FileErrorCode = 15


def filterModule(backendName: Optional[str], moduleList: List['ModuleImplement']) \
        -> List['ModuleImplement']:
    if backendName is None:
        return moduleList

    if backendName in [
        BackendName.LocalBaiduSim2.value,
        
    ]:
        return _filterSimulator(backendName, moduleList)
    
    else:
        return moduleList


def _filterSimulator(backendName: str, moduleList: List['ModuleImplement']) -> List['ModuleImplement']:
    unrollCircuitModule = None  # type: Optional[UnrollCircuitModule]
    compressGateModule = None  # type: Optional[CompressGateModule]
    ret = []  # type: List['ModuleImplement']
    for module in moduleList:
        
        if module.__class__.__name__ == 'UnrollCircuitModule':
            unrollCircuitModule = module
        elif module.__class__.__name__ == 'CompressGateModule':
            compressGateModule = module
        elif not module.disable:
            ret.append(module)
    if unrollCircuitModule is not None:
        if not unrollCircuitModule.disable:
            ret.append(unrollCircuitModule)
    else:
        ret.append(UnrollCircuitModule())
    if backendName not in [
        
    ]:
        if compressGateModule is not None:
            if not compressGateModule.disable:
                ret.append(compressGateModule)
        else:
            ret.append(CompressGateModule())
    return ret



