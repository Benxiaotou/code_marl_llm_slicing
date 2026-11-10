"""Network guard inherited by offline-test subprocesses via PYTHONPATH."""
import ipaddress
import socket

_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def _local(address):
    if not isinstance(address, tuple):
        return True
    host = address[0]
    if host in ('localhost', b'localhost'):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def connect(sock, address):
    if not _local(address):
        raise RuntimeError('Offline tests blocked a non-loopback connection')
    return _connect(sock, address)


def connect_ex(sock, address):
    if not _local(address):
        raise RuntimeError('Offline tests blocked a non-loopback connection')
    return _connect_ex(sock, address)


def getaddrinfo(host, port, *args, **kwargs):
    if host is not None and not _local((host, port)):
        raise RuntimeError('Offline tests blocked non-loopback DNS resolution')
    return _getaddrinfo(host, port, *args, **kwargs)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
