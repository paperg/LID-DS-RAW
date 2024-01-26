

1. res 返回操作大小

['write', 'mmap', 'brk', 'writev', 'read', 'sendto', 'recvfrom', 'lseek', 'pwrite', 'pread' 'sendmsg', 'recvmsg', 'sendfile'] 特殊处理
write : res 成功写入字节数 data=... 数据
writev : res 成功写入字节数 data=... 数据
read : res 成功读取字节数 data=... 数据

recvfrom < res=7240 data=JSFQUy1BZG9iZS0zLjAgRVBTRi0zLjAKJSVDcmVhdG9yOiAoSW1hZ2VNYWdpY2spCiUlVGl0bGU6IChub3JtYWwvaW1hZ2VzL2ltYWdlXzA= tuple=NULL
write < res=15101 data=JSFQUy1BZG9iZS0zLjAgRVBTRi0zLjAKJSVDcmVhdG9yOiAoSW1hZ2VNYWdpY2spCiUlVGl0bGU6IChub3JtYWwvaW1hZ2VzL2ltYWdlXzA=
writev < res=8265 data=IC9wc3RvZWRpdC5jb3B5cmlnaHQgKENvcHlyaWdodCBcKENcKSAxOTkzIC0gMjAxNCBXb2xmZ2FuZyBHbHVueikgZGVmIAogL3BzdG9lZGk=
pwrite < res=512 data=AAAAAAAAAAAAAAAAFNIPVQAAAAcAAAIAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=
sendto < res=98 data=SFRUUC8xLjAgMjAxIENyZWF0ZWQNClNlcnZlcjogU2ltcGxlSFRUUC8wLjYgUHl0aG9uLzMuNS4yDQpEYXRlOiBXZWQsIDA4IFNlcCAyMDI=
read < res=0 data=
sendmsg < res=249 data=FgMBAPQBAADwAwNhOy0kw9Kh/8fYyfepvZyBt1s6auetASGYSmRYG+3hRgAAcsAswIfMqcCtwArAJMBzwCvAhsCswAnAI8BywAjAMMCLzKg=
recvmsg < res=164 size=164 data=TAAAABQAAgCRWjhhagAAAAIIgP4BAAAACAABAH8AAAEIAAIAfwAAAQcAAwBsbwAACAAIAIAAAAAUAAYA//////////8QAH0AEAB9AFgAAAA= tuple=NULL
pread < res=16 data=AAAABAAAAAcAAAAAAAAAAA==
sendfile < res=21080 offset=21080

2. res 返回地址范围

lseek : lseek < res=0

brk < res=1813000 vm_size=332 vm_rss=4 vm_swap=0
mmap : mmap < res=7FBD92DEA000 vm_size=112392
munmap < res=0 vm_size=17836 vm_rss=1852 vm_swap=0


3. 返回结果

setgid > gid=33(www-data)
setgid < res=0
setuid > uid=33(www-data)
setuid < res=0

clone < res=0 exe=puma 3.12.0 (tcp://0.0.0.0:3000 args= tid=2724(puma 003) pid=1962(ruby) ptid=1892(start_rails.sh) cwd= fdlimit=1048576 pgft_maj=0 pgft_min=0 vm_size=1774720 vm_rss=129436 vm_swap=0 comm=puma 003 cgroups=Y3B1c2V0PS9kb2NrZXIvNGZmNDM1OWEzM2EwYzNiNGQ4NTdjYmFkZjQ1MTEyZjBhN2RjYjBkODdkOGYwZWEzZmI5MjQyODBiZTNiZDNjYgBjcHU9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAGNwdWFjY3Q9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAGlvPS9kb2NrZXIvNGZmNDM1OWEzM2EwYzNiNGQ4NTdjYmFkZjQ1MTEyZjBhN2RjYjBkODdkOGYwZWEzZmI5MjQyODBiZTNiZDNjYgBtZW1vcnk9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAGRldmljZXM9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAGZyZWV6ZXI9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAG5ldF9jbHM9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAHBlcmZfZXZlbnQ9L2RvY2tlci80ZmY0MzU5YTMzYTBjM2I0ZDg1N2NiYWRmNDUxMTJmMGE3ZGNiMGQ4N2Q4ZjBlYTNmYjkyNDI4MGJlM2JkM2NiAG5ldF9wcmlvPS9kb2NrZXIvNGZmNDM1OWEzM2EwYzNiNGQ4NTdjYmFkZjQ1MTEyZjBhN2RjYjBkODdkOGYwZWEzZmI5MjQyODBiZTNiZDNjYgBodWdldGxiPS9kb2NrZXIvNGZmNDM1OWEzM2EwYzNiNGQ4NTdjYmFkZjQ1MTEyZjBhN2RjYjBkODdkOGYwZWEzZmI5MjQyODBiZTNiZDNjYgBwaWRzPS9kb2NrZXIvNGZmNDM1OWEzM2EwYzNiNGQ4NTdjYmFkZjQ1MTEyZjBhN2RjYjBkODdkOGYwZWEzZmI5MjQyODBiZTNiZDNjYgByZG1hPS8A flags=578861571(CLONE_FILES|CLONE_FS|CLONE_PARENT_SETTID|CLONE_SIGHAND|CLONE_SYSVSEM|CLONE_THREAD|CLONE_VM|CLONE_CHILD_CLEARTID|CLONE_SETTLS) uid=0 gid=0 vtid=28 vpid=8


close > fd=5(<f>/service/upload/image_0456.eps)
close < res=0

pipe >
pipe < res=0 fd1=5(<p>) fd2=6(<p>) ino=102743167

access < res=-2(ENOENT) name=/etc/ld.so.nohwcap

poll > fds=3:41 timeout=500
poll < res=1 fds=3:41

munmap > addr=7EFDC7906000 length=14277
munmap < res=0 vm_size=17836 vm_rss=1852 vm_swap=0

kill > pid=50 sig=13(SIGPIPE)
kill < res=0

getrlimit > resource=3(RLIMIT_STACK)
getrlimit < res=0 cur=8388608 max=-1

futex > addr=7F02E96D7740 op=129(FUTEX_PRIVATE_FLAG|FUTEX_WAKE) val=2147483647
futex < res=0

connect > fd=5(<u>)
connect < res=-2(ENOENT) tuple=0->ffff94ea2a949000 /var/run/nscd/socket

bind > fd=5(<n>)
bind < res=0 addr=NULL

prlimit > pid=0 resource=6(RLIMIT_NPROC)
prlimit < res=0 newcur=-1 newmax=-1 oldcur=-1 oldmax=-1
setpgid > pid=0 pgid=50
setpgid < res=0

select >
select < res=1

flock > fd=10(<f>/root/.wget-hsts) operation=2(LOCK_EX)
flock < res=0

getsockopt >
getsockopt < res=0 fd=45(<4t>172.21.0.10:48454->172.21.0.26:3000) level=2(SOL_TCP) optname=0(UNKNOWN) val=AQAAAAAHdwDgHAMAQJwAAKgFAADLAgAAAAAAAAAAAAAAAAAAAAAAAAAAAACwAwAAAAAAALADAACwAwAA3AUAAEz6AAAWAAAACwAAAP///38KAAAAqAUAAAMAAADoAwAACDkAAAAAAAB0HXZOAAAAAP//////////AAAAAAAAAADLAgAAAAAAAAEAAAADAAAAAAAAABYAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPsAAA== optlen=232

setsockopt >
setsockopt < res=0 fd=45(<4t>172.21.0.10:48454->172.21.0.26:3000) level=2(SOL_TCP) optname=0(UNKNOWN) val=AQAAAA== optlen=4

lstat >
lstat < res=0 path=/usr

tgkill > pid=1(systemd) tid=13 sig=26(SIGVTALRM)
tgkill < res=0

nanosleep > interval=0(0s)
nanosleep < res=0

epoll_wait < res=1
epoll_wait > maxevents=512

semop > semid=0
semop < res=0 nsops=1 sem_num_0=0 sem_op_0=1 sem_flg_0=2(SEM_UNDO) sem_num_1=0 sem_op_1=0 sem_flg_1=0

4. 涉及文件

- 需要 > 的系统调用

RET:
write > fd=5(<f>/service/upload/image_0456.eps) size=15101
writev > fd=5(<f>/tmp/psindWfZyH) size=8265
pwrite > fd=81(<f>/usr/src/blog/db/development.sqlite3-journal) size=512 pos=0
pread > fd=27(<f>/usr/src/blog/db/development.sqlite3) size=16 pos=24
close > fd=5(<f>/service/upload/image_0456.eps)
getdents64 > fd=5(<f>/proc/self/fd)
getdents > fd=3(<d>/usr/bin)
ioctl > fd=5(<f>/service/upload/image_0456.eps) request=5401 argument=7FFFE91146B0
execve > filename=/usr/local/sbin/pstoedit
lseek > fd=5(<f>/service/upload/image_0456.eps) offset=0 whence=1(SEEK_CUR)
fcntl > fd=1(<f>/dev/pts/0) cmd=1(F_DUPFD)
dup > fd=2(<f>/dev/pts/0)
flock > fd=10(<f>/root/.wget-hsts) operation=2(LOCK_EX)
fstat > fd=3(<f>/etc/ld.so.cache)

ARGS1:
? sendfile > out_fd=3(<4t>172.24.0.6:48956->172.24.0.7:80) in_fd=10(<f>/etc/nginx/html/index.html) offset=0 size=21080

ARGS4:
mmap > addr=0 length=14277 prot=1(PROT_READ) flags=2(MAP_PRIVATE) fd=3(<f>/etc/ld.so.cache) offset=0



RET:
getdents64 > fd=5(<f>/proc/self/fd)
getdents64 < res=216
getdents > fd=3(<d>/usr/bin)
getdents < res=9416
fcntl > fd=1(<f>/dev/pts/0) cmd=1(F_DUPFD)
fcntl < res=10(<f>/dev/pts/0)
dup > fd=2(<f>/dev/pts/0)
dup < res=1(<f>/dev/pts/0)

openat >
openat < fd=46(<d>/usr/src/blog/db/migrate) dirfd=-100(AT_FDCWD) name=db/migrate(/usr/src/blog/db/migrate) flags=13313(O_DIRECTORY|O_RDONLY|O_CLOEXEC|O_TMPFILE) mode=0 dev=300072

ARGS1:
lstat >
lstat < res=0 path=/usr
chmod >
chmod < res=0 filename=../../hackable/uploads/tmpytdcy79w(/var/www/html/hackable/uploads/tmpytdcy79w) mode=0644(S_IROTH|S_IRGRP|S_IWUSR|S_IRUSR)
open >
open < fd=-2(ENOENT) name=/etc/localtime flags=4097(O_RDONLY|O_CLOEXEC) mode=0 dev=0
clone >
clone < res=9 exe=python3 args=Li4vSW1hZ2VDb252ZXJ0ZXIucHkA tid=3793567(python3) pid=3793567(python3) ptid=3793476(bash) cwd= fdlimit=1048576 pgft_maj=0 pgft_min=2592 vm_size=52836 vm_rss=16576 vm_swap=0 comm=python3 cgroups=Y3B1c2V0PS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABjcHU9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGNwdWFjY3Q9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGlvPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABtZW1vcnk9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGRldmljZXM9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGZyZWV6ZXI9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAG5ldF9jbHM9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAHBlcmZfZXZlbnQ9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAG5ldF9wcmlvPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABodWdldGxiPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABwaWRzPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZAByZG1hPS8A flags=562036736(CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID) uid=0 gid=0 vtid=8 vpid=8

execve > filename=/usr/local/sbin/pstoedit
execve < res=-2(ENOENT) exe=pstoedit args=LWYAcGxvdC1zdmcAaW1hZ2VfMDE3Ni5lcHMAaW1hZ2VfMDE3Ni5lcHMuc3ZnAA== tid=3794094(python3) pid=3794094(python3) ptid=3793567(python3) cwd= fdlimit=1048576 pgft_maj=0 pgft_min=26 vm_size=52836 vm_rss=9580 vm_swap=0 comm=python3 cgroups=Y3B1c2V0PS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABjcHU9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGNwdWFjY3Q9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGlvPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABtZW1vcnk9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGRldmljZXM9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAGZyZWV6ZXI9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAG5ldF9jbHM9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAHBlcmZfZXZlbnQ9L2RvY2tlci9lZGY1NTE5YWIwNjUxZTNmNTlkN2Y5OGMzZmFiYzYxNDlkMjk2ODk5NzUxNzFjMmY4MDBmMTdhODdhYTExNDJkAG5ldF9wcmlvPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABodWdldGxiPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZABwaWRzPS9kb2NrZXIvZWRmNTUxOWFiMDY1MWUzZjU5ZDdmOThjM2ZhYmM2MTQ5ZDI5Njg5OTc1MTcxYzJmODAwZjE3YTg3YWExMTQyZAByZG1hPS8A env=SE9TVE5BTUU9ZWRmNTUxOWFiMDY1AFRFUk09eHRlcm0AUEFUSD0vdXNyL2xvY2FsL3NiaW46L3Vzci9sb2NhbC9iaW46L3Vzci9zYmluOi91c3IvYmluOi9zYmluOi9iaW4AUFdEPS9zZXJ2aWNlL3VwbG9hZABTSExWTD0xAEhPTUU9L3Jvb3QAT0xEUFdEPS8AXz0vdXNyL2Jpbi9weXRob24zAA== tty=34816 pgid=1(systemd) loginuid=-1
vfork >
vfork < res=0 exe=/usr/local/openjdk-8/bin/java args=LURqYXZhLnV0aWwubG9nZ2luZy5jb25maWcuZmlsZT0vdXNyL2xvY2FsL3RvbWNhdC9jb25mL2xvZ2dpbmcucHJvcGVydGllcwAtRGphdmEudXRpbC5sb2dnaW5nLm1hbmFnZXI9b3JnLmFwYWNoZS5qdWxpLkNsYXNzTG9hZGVyTG9nTWFuYWdlcgAtRGpkay50bHMuZXBoZW1lcmFsREhLZXlTaXplPTIwNDgALURqYXZhLnByb3RvY29sLmhhbmRsZXIucGtncz1vcmcuYXBhY2hlLmNhdGFsaW5hLndlYnJlc291cmNlcwAtRG9yZy5hcGFjaGUuY2F0YWxpbmEuc2VjdXJpdHkuU2VjdXJpdHlMaXN0ZW5lci5VTUFTSz0wMDI3AC1EaWdub3JlLmVuZG9yc2VkLmRpcnM9AC1jbGFzc3BhdGgAL3Vzci9sb2NhbC90b21jYXQvYmluL2Jvb3RzdHJhcC5qYXI6L3Vzci9sb2NhbC90b21jYXQvYmluL3RvbWNhdC1qdWxpLmphcgAtRGNhdGFsaW5hLmJhc2U9L3Vzci9sb2NhbC90b21jYXQALURjYXRhbGluYS5ob21lPS91c3IvbG9jYWwvdG9tY2F0AC1EamF2YS5pby50bXBkaXI9L3Vzci9sb2NhbC90b21jYXQvdGVtcABvcmcuYXBhY2hlLmNhdGFsaW5hLnN0YXJ0dXAuQm9vdHN0cmFwAHN0YXJ0AA== tid=25461(http-nio-8080-e) pid=25461(http-nio-8080-e) ptid=21655(http-nio-8080-e) cwd= fdlimit=1048576 pgft_maj=0 pgft_min=0 vm_size=12347460 vm_rss=520616 vm_swap=0 comm=http-nio-8080-e cgroups=Y3B1c2V0PS9kb2NrZXIvNDFmN2ZlZTAwMjU4NDcxZGMzODlmN2E0YzE3N2Y3ZDk3Nzk3YzUwOTNkZWU2NDFjMzhhZDJlY2FmMDA0MDU4YQBjcHU9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAGNwdWFjY3Q9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAGlvPS9kb2NrZXIvNDFmN2ZlZTAwMjU4NDcxZGMzODlmN2E0YzE3N2Y3ZDk3Nzk3YzUwOTNkZWU2NDFjMzhhZDJlY2FmMDA0MDU4YQBtZW1vcnk9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAGRldmljZXM9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAGZyZWV6ZXI9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAG5ldF9jbHM9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAHBlcmZfZXZlbnQ9L2RvY2tlci80MWY3ZmVlMDAyNTg0NzFkYzM4OWY3YTRjMTc3ZjdkOTc3OTdjNTA5M2RlZTY0MWMzOGFkMmVjYWYwMDQwNThhAG5ldF9wcmlvPS9kb2NrZXIvNDFmN2ZlZTAwMjU4NDcxZGMzODlmN2E0YzE3N2Y3ZDk3Nzk3YzUwOTNkZWU2NDFjMzhhZDJlY2FmMDA0MDU4YQBodWdldGxiPS9kb2NrZXIvNDFmN2ZlZTAwMjU4NDcxZGMzODlmN2E0YzE3N2Y3ZDk3Nzk3YzUwOTNkZWU2NDFjMzhhZDJlY2FmMDA0MDU4YQBwaWRzPS9kb2NrZXIvNDFmN2ZlZTAwMjU4NDcxZGMzODlmN2E0YzE3N2Y3ZDk3Nzk3YzUwOTNkZWU2NDFjMzhhZDJlY2FmMDA0MDU4YQByZG1hPS8A flags=536870912 uid=0 gid=0 vtid=49 vpid=49

access > mode=0(F_OK)
access < res=-2(ENOENT) name=/etc/ld.so.nohwcap

stat >
stat < res=-2(ENOENT) path=image_0456.eps(/service/upload/image_0456.eps)

unlink >
unlink < res=0 path=/tmp/psindWfZyH

getcwd >
getcwd < res=5 path=/app

mkdir > mode=0
mkdir < res=0 path=/usr/src/blog/tmp/cache/assets/sprockets/v3.0/UK

rename >
rename < res=0 oldpath=/usr/src/blog/tmp/cache/assets/sprockets/v3.0/UK/UKg9qCUMs6fqi0fGic2LHIVhmk1DulaiZjEltxJ2lQs.cache.47455009286920.1.587966 newpath=/usr/src/blog/tmp/cache/assets/sprockets/v3.0/UK/UKg9qCUMs6fqi0fGic2LHIVhmk1DulaiZjEltxJ2lQs.cache

chdir >
chdir < res=0 path=/var/www/html


5. 其他：
mode: access >
size: sendto > write > read >
length: mmap >
prot:  mmap >
flags: mmap > accept >
cmd： fcntl >


accept > flags=0
accept < fd=4(<4t>192.168.240.5:57566->192.168.240.15:8000) tuple=192.168.240.5:57566->192.168.240.15:8000 queuepct=0 queuelen=0 queuemax=5



6. None:
getuid >
getuid < uid=0(root)
getgid >
getgid < gid=0(root)
geteuid >
geteuid < euid=0(root)
getegid >
getegid < egid=0(root)

procexit > status=0

socket > domain=1(AF_LOCAL) type=526337 proto=0
socket < fd=5(<u>)

setgroups >
setgroups <

fadvise64 >
fadvise64 <

1631783052873213116 1001 1540723 node 1540754 <unknown> >
1631783052873216910 1001 1540723 node 1540754 <unknown> <
