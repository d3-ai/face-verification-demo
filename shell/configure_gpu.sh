#!/bin/bash
mig_enabled=1
sudo nvidia-smi -pm 1
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -i 0 -cgi 1g.5gb,1g.5gb,1g.5gb,4g.20gb -C
sudo nvidia-smi mig -i 1 -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb, -C
nvidia-smi -L | grep MIG | awk '{print $6}' | sed -e 's/)/,/g' | sed -z 's/\n//g'> ./.devcontainer/.gpu_uuid
echo "" >> ./.devcontainer/.gpu_uuid