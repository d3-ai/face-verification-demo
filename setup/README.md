# How to setup raspberry pi cluster

1. Install `ansible`

```bash
# apt install ansible sshpass
```

2. Burn Ubuntu server 20.04 image for raspberry pi to microSD cards and boot

3. Edit hosts.yml to point raspberry pi machines

```yaml
all:
  children:
    pi-cluster:
      vars:
        ansible_user: ubuntu
        ansible_password: ubuntu
        ansible_become_password: ubuntu
      hosts:
        192.168.20.[20:29]:
```

4. Run `ansible-playbook` to setup raspberry pi cluster

```bash
$ ansible-playbook -i hosts.yml pi-cluster.yml
```
