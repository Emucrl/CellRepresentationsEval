source /root/anaconda3/etc/profile.d/conda.sh
conda init
conda create -n graphs python=3.10
conda activate graphs
pip install -r requirements.txt
pre-commit install
echo "PYTHONPATH=${PYTHONPATH}:${SCRIPTS_DIR}" > .env
echo "export PATH=/root/anaconda3/envs/graphs/bin:$PATH" >> ~/.bashrc
