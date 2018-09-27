echo 'Updating your packages'
sudo apt-get update && sudo apt-get upgrade -y
echo 'Installing Python, the Python Package Installer and software to downlaod code from github'
sudo apt-get install python3 python3-pip -y
echo 'Downloading code from Github'
git clone https://github.com/qwertpi/catvdogcnn.git catdogcnn
cd catdogcnn
echo 'Installing the requried python libaries'
sudo pip3 install -r requirements.txt
sudo apt-get install libhdf5-serial-dev
