conda install -c conda-forge pyproj
conda install -c conda-forge pyepsg


export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost

git clone https://github.com/esiivola/syke-machine-learning-course

cd syke-machine-learning-course

git stash
git pull
git stash pop
