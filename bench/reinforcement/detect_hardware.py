"""Hardware detection for benchmark documentation."""

import subprocess
import platform


def get_hardware_info() -> dict:
    """Detect Apple Silicon hardware specifications."""
    info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
    
    # Try to get detailed Apple Silicon info
    try:
        result = subprocess.run(
            ['system_profiler', 'SPHardwareDataType'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('Chip:'):
                    info['chip'] = line.split(':', 1)[1].strip()
                elif line.startswith('Memory:'):
                    info['memory'] = line.split(':', 1)[1].strip()
                elif line.startswith('Number of CPUs:'):
                    info['cpus'] = line.split(':', 1)[1].strip()
                    
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Could not get detailed hardware info: {e}")
    
    return info


def format_hardware_string(info: dict) -> str:
    """Format hardware info for README."""
    parts = []
    
    if 'chip' in info:
        parts.append(info['chip'])
    elif 'processor' in info and info['processor']:
        parts.append(info['processor'])
    
    if 'memory' in info:
        parts.append(info['memory'])
    
    if 'cpus' in info:
        parts.append(f"{info['cpus']} CPU")
    
    return ', '.join(parts) if parts else 'Unknown hardware'


if __name__ == '__main__':
    info = get_hardware_info()
    print("Hardware Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nFormatted for README: {format_hardware_string(info)}")
