import platform

if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl