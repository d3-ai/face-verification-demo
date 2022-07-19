#!/bin/bash

uid=$(id -u $USER)
gid=$(id -g $USER)

uname=$(id -un $USER) 
gname=$(id -gn $USER)

if getent group "$gid" > /dev/null 2>&1; then
    echo "GROUP_ID '$gid' already exists."
else
    echo "GROUP_ID '$gid' does NOT exist. So execute [groupadd -g \$GROUP_ID \$GROUP_NAME]."
    groupadd -g $gid $gname
fi

if getent passwd "$uid" > /dev/null 2>&1; then
    echo "USER_ID '$uid' already exists."
else
    echo "USER_ID '$uid' does NOT exist. So execute [useradd -m -s /bin/bash -u \$USER_ID -g \$GROUP_ID \$USER_NAME]."
    useradd -m -s /bin/bash -u $uid -g $gid $uname
fi



exec setpriv --reuid=$uid --regid=$gid --init-groups "$@"