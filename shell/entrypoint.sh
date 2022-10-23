#!/bin/bash

uid=$(id -u $USER)
gid=$(id -g $USER)

uname=$(id -un $USER) 
gname=$(id -gn $USER)

gid_exist=0
uid_exist=0

if getent group "$gid" > /dev/null 2>&1; then
    echo "GROUP_ID '$gid' already exists."
    gid_exist=1
else
    echo "GROUP_ID '$gid' does NOT exist. So execute [groupadd -g \$GROUP_ID \$GROUP_NAME]."
    groupadd -g $gid $gname
fi

if getent passwd "$uid" > /dev/null 2>&1; then
    echo "USER_ID '$uid' already exists."
    uid_exist=1
else
    echo "USER_ID '$uid' does NOT exist. So execute [useradd -m -s /bin/bash -u \$USER_ID -g \$GROUP_ID \$USER_NAME]."
    useradd -m -s /bin/bash -u $uid -g $gid $uname
fi

if [ "$gid_exist" ] || [ "$uid_exist" ]; then
    exec "$@"
else
    exec setpriv --reuid=$uid --regid=$gid --init-groups "$@"
fi