#*******************************************************************************#
#       Copyright (c) 2022 TraceUp All Rights Reserved.			                #
# 	    Author: Kevin Foyet                                                     #
#       Date: 09/22/2022                                                        #
#*******************************************************************************#


# This script is used to ssh into cases and get their logs files.

from logging import exception, shutdown
from tkinter import messagebox
from typing import Type
import paramiko
import ipaddress
import sys
import os
import Get_Case_Logs
from scp import SCPClient


command = "df"

# IP Address, username, and password need to create a connection
host = input("Enter a list of IP ADDRESSES seperated by a \',\': ")
ip_addresses = host.split(",")
print(ip_addresses)
username = "trace"
password = "trace"

# Checks if IP Address is valid if not then it throws an exception
for ip_address in ip_addresses:
    try:
        ip = ipaddress.ip_address(ip_address)
    except ValueError:
        print("IP address {} is not valid".format(ip_address)) 
        print("Unidentified Host, Connection failed")
        raise SystemExit

    # Creates a connection to SSH session

    client = paramiko.client.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip_address, username=username, password=password)
    _stdin, _stdout,_stderr = client.exec_command("df")
    print(_stdout.read().decode())

    # Gets the file from ssh and sends it to tarfile in your local directory
    ftp_client = client.open_sftp()
    dir_list = ftp_client.listdir('/home/trace/tarfiles/')
    for i in dir_list:
        if "data" in i:
            ftp_client.get('/home/trace/tarfiles/'+i,f'/Users/{os.getlogin()}/Documents/GitHub/hardware_scripts-main/Case/Tar_Gz_Files/'+i)

    # Runs the Script  
    Get_Case_Logs.main(sys.argv[1:])

ftp_client.close()