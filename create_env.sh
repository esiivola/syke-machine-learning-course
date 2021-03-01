export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost

cd ~

git clone https://github.com/esiivola/syke-machine-learning-course

cd ./syke-machine-learning-course
pip install -r requirements.txt
#conda install -c conda-forge cartopy=0.18.0
