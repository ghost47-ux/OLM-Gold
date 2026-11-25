"""Hardware and software detection used for reporting and device selection."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from dataclasses import asdict, dataclass


@dataclass
class EnvironmentReport:
    os: str
    python: str
    cpu: str
    physical_cores: int | None
    logical_cores: int
    ram_gb: float | None
    torch: str | None
    cuda_available: bool
    cuda_device: str | None
    device: str

    def as_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        rows = [
            ('Operating system', self.os),
            ('Python', self.python),
            ('CPU', self.cpu),
            ('Physical cores', self.physical_cores or 'unknown'),
            ('Logical cores', self.logical_cores),
            ('RAM (GB)', f'{self.ram_gb:.1f}' if self.ram_gb else 'unknown'),
            ('PyTorch', self.torch or 'not installed'),
            ('CUDA available', self.cuda_available),
            ('CUDA device', self.cuda_device or '-'),
            ('Selected device', self.device),
        ]
        width = max(len(k) for k, _ in rows)
        return '\n'.join(f'{k:<{width}} : {v}' for k, v in rows)


def _total_ram_bytes() -> int | None:
    try:
        if sys.platform == 'win32':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None
        with open('/proc/meminfo', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def detect_environment() -> EnvironmentReport:
    torch_version = None
    cuda = False
    cuda_device = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        if cuda:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass
    ram = _total_ram_bytes()
    return EnvironmentReport(
        os=f'{platform.system()} {platform.release()} ({platform.version()})',
        python=platform.python_version(),
        cpu=platform.processor() or platform.machine(),
        physical_cores=_physical_cores(),
        logical_cores=os.cpu_count() or 1,
        ram_gb=(ram / 1024**3) if ram else None,
        torch=torch_version,
        cuda_available=cuda,
        cuda_device=cuda_device,
        device='cuda' if cuda else 'cpu',
    )


def resolve_device(requested: str = 'auto') -> str:
    import torch

    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA device is available')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError(f'unsupported device {requested!r}')
    return requested
