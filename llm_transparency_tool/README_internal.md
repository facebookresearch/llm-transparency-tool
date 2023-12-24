# LLM Viewer

# Using a running instance

1. On your laptop, ssh to Igor's devfair with port forwarding:

```bash
HOST=100.96.183.109; PORT=8502; ssh $HOST -J $USER@ash-fairjmp102 -L $PORT:$HOST:$PORT
```

2. In your browser open `localhost:8502`.

# Installation

```bash
conda create -n llm_transparency_tool python=3.10
conda activate llm_transparency_tool

conda install yarn

git clone git@github.com:fairinternal/transparent-llms.git
cd ./transparent-llms
pip install -e .
pip install -r ./requirements_contributions.txt

# When running on FAIR cluster, you may encounter old nvidia drivers, but downgrading
# the pytorch helps
pip install torch==2.0

cd ./llm_transparency_tool/components/frontend
yarn install
yarn build

cd ../../../
streamlit run ./llm_transparency_tool/server/app.py -- ./llm_transparency_tool/config/local.json
```

Sometimes `yarn start` and `streamlit run` may complain about the reaching the limit
of watch files: `# OSError(errno.ENOSPC, "inotify watch limit reached")`.
It can help to shut down the VS Code and stop its servers because they watch files too:

```bash
ps aux | grep vscode | awk '{print $2}' | xargs -I % kill -9 %
```

If you then run the command that you want (`yarn` or `streamlit`), it will watch the
necessary files and you can start VS Code again.
