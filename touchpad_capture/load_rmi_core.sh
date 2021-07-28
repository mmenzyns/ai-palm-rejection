#!/usr/bin/env bash
# Launch with sudo permissions
sudo rmmod rmi_smbus
sudo rmmod rmi_core
sudo insmod 5.9.0-rc5_rmi_core.ko debug_flags=1
sudo su -c "echo 1 > /sys/module/psmouse/parameters/synaptics_intertouch"
sudo su -c "echo -n rescan > /sys/bus/serio/devices/serio1/drvctl"


