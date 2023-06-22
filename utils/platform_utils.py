import socket
COMPUTER = "127.0.1.1" #本地电脑的ip
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

if COMPUTER in ip_address:
    USE_COMPUTER = True
else:
    USE_COMPUTER =False

WATCH = False #控制在本地想不想看
