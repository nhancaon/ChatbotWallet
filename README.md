sudo apt-get update
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda create -n chatbot python=3.10
pip install -r requirements.txt
sudo app install -y nginx
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
server {
listen 80;
server_name "Your server name";
location / {
proxy_pass http://127.0.0.1:8000;
}
}
sudo service nginx restart
sudo nano /etc/systemd/system/uvicorn.service
[Unit]
Description=Uvicorn instance to serve FastAPI
After=network.target
[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ChatbotWallet
ExecStartPre=/bin/bash -c "source /home/ubuntu/miniconda3/bin/activate chatbot"
ExecStart=/bin/bash -c "exec uvicorn main:app --host 0.0.0.0 --port 8000"
Restart=always
Environment="PYTHONPATH=/home/ubuntu/ChatbotWallet"
Environment="CONDA_EXE=/home/ubuntu/miniconda3/bin/conda"
Environment="CONDA_PREFIX=/home/ubuntu/miniconda3/envs/chatbot"
Environment="CONDA_PYTHON_EXE=/home/ubuntu/miniconda3/envs/chatbot/bin/python"
Environment="PATH=/home/ubuntu/miniconda3/envs/chatbot/bin:$PATH"
[Install]
WantedBy=multi-user.target
sudo systemctl start uvicorn
sudo systemctl enable uvicorn
sudo systemctl daemon-reload
sudo systemctl restart uvicorn
sudo systemctl status uvicorn
