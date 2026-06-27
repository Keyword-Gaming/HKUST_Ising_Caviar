# Remote Development Setup

This tutorial explains how to log in to the remote machines and set up a development environment for the internship.

We will use the TACC cluster through SSH. Most experiments should run on the GPU machines, not on the jump host.

## 1. Access The Remote Machine

The TACC cluster is a group of remote machines. Because these machines are protected, we connect to them using SSH.

SSH is a secure way to control a remote computer from your own laptop.

### Step 1: Generate Your SSH Key

On your local computer, open Terminal and run:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

When it asks where to save the key, you can press Enter to use the default location.

Then display your public key:

```bash
cat ~/.ssh/id_rsa.pub
```

Copy the full output. It should start with `ssh-rsa`.

Send this public key to your instructor so your account can be created.

Important:

- The public key, `id_rsa.pub`, can be shared.
- The private key, `id_rsa`, must stay on your own computer.
- Do not send your private key to anyone.

### Step 2: Configure SSH

For security reasons, you cannot connect directly to the GPU machines. You first connect through a jump host.

Create or edit this file on your local computer:

```bash
~/.ssh/config
```

Add the following configuration:

```text
Host cpu01
    HostName gw.tacc.ust.hk
    User [your_username]

Host gpu13
    HostName gpu13
    User [your_username]
    ProxyJump cpu01

Host gpu14
    HostName gpu14
    User [your_username]
    ProxyJump cpu01
```

Replace `[your_username]` with your assigned username.

The username is usually based on your first name and last name.

### Step 3: Understand The Machines

There are two types of machines:

- `cpu01`: the jump host. Treat this as the lobby. Do not run code, install packages, or start experiments here.
- `gpu13` / `gpu14`: GPU machines. These are where you should run experiments and development work.

To connect to a GPU machine, run:

```bash
ssh gpu13
```

or:

```bash
ssh gpu14
```

### Step 4: Change Your Password

After logging in for the first time, change your password:

```bash
passwd
```

The initial password is:

```text
{username}123
```

Choose a new password that includes letters, numbers, and special characters.

## 2. Recommended Environment: Docker

We recommend Docker for this internship.

Docker creates an isolated environment with the required libraries and tools. This helps avoid the common problem where code works on one person's computer but fails on another person's computer.

### Step 1: Prepare Your Workspace

Run these commands on your assigned GPU machine:

```bash
mkdir -p ~/work
mkdir -p /mnt/nfs/[your_username]/.cache
```

Replace `[your_username]` with your assigned username.

Use:

- `~/work` for your project code.
- `/mnt/nfs/[your_username]` for larger files, such as models and datasets.

### Step 2: Pull The Docker Image

On your assigned GPU machine, download the project image:

```bash
docker pull vllm/vllm-openai:v0.9.2
```

If Docker asks you to log in, create a Docker Hub account at <https://hub.docker.com/> and run:

```bash
docker login
```

Then run the `docker pull` command again.

### Step 3: Start Your Container

Start a Docker container with:

```bash
docker run --name [your_username]_vllm_dev -it \
    --gpus all \
    --volume /home/[your_username]/work:/usr/wkspace \
    --volume /mnt/nfs/[your_username]/.cache:/usr/data \
    --network=host \
    --ipc=host \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.9.2
```

Replace every `[your_username]` with your assigned username.

After entering the container, your project folder will be available at:

```bash
/usr/wkspace
```

Your shared cache/data folder will be available at:

```bash
/usr/data
```

### Docker Rules

- Always give your container a clear name, such as `[your_username]_vllm_dev`.
- Do not map the whole system disk into Docker.
- Use `/mnt/nfs/[your_username]` for large files.
- Do not run heavy experiments on `cpu01`.

### Common Docker Commands

Check all containers:

```bash
docker ps -a
```

Stop a container:

```bash
docker stop [container_name]
```

Resume a stopped container:

```bash
docker start -i [container_name]
```

Delete a container:

```bash
docker rm -f [container_name]
```

## 3. Alternative Environment: Conda

Docker is recommended. If Docker does not work for your account or machine, you can use Conda instead.

Conda manages separate Python environments. This keeps packages for different projects from interfering with each other.

### Step 1: Install Miniconda

Run these commands on your assigned GPU machine:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

After installation, close your SSH session and reconnect:

```bash
exit
ssh gpu13
```

or:

```bash
ssh gpu14
```

### Step 2: Create A Conda Environment

Create a new environment:

```bash
conda create -n vllm python=3.10 -y
```

Activate it:

```bash
conda activate vllm
```

When the environment is active, your terminal prompt should show:

```text
(vllm)
```

Install Python packages only after activating the correct environment.

## 4. Quick Check

After logging in to a GPU machine, check that GPU access works:

```bash
nvidia-smi
```

If the command shows GPU information, you are on a GPU machine.

If `nvidia-smi` fails, check that:

- You are connected to `gpu13` or `gpu14`.
- You are not running commands on `cpu01`.
- Your account has been created correctly.

## 5. What To Send To The Instructor

Before your account is ready, send your instructor:

```text
Your public SSH key from ~/.ssh/id_rsa.pub
```

Do not send:

```text
~/.ssh/id_rsa
```

That is your private key.
