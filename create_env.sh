export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost

cd ~

git clone https://github.com/esiivola/syke-machine-learning-course

cd ./syke-machine-learning-course
apt-get install libproj-dev proj-data proj-bin  
apt-get install libgeos-dev  

pip install -r requirements.txt
#conda install -c conda-forge cartopy=0.18.0
