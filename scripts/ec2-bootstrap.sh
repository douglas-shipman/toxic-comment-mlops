#!/bin/bash
set -e

dnf update -y
dnf install -y docker git

systemctl enable docker
systemctl start docker

usermod -aG docker ec2-user

cd /opt
git clone https://github.com/douglas-shipman/toxic-comment-mlops.git
chown -R ec2-user:ec2-user /opt/toxic-comment-mlops