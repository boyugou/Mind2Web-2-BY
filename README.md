# Mind2Web 2

Mind2Web 2 is an evaluation framework for agentic search capabilities, featuring Agent-as-a-Judge methodology for comprehensive assessment of web automation agents.

<div align="center">
  <img src="./assets/mind2web2_overview.jpg" alt="Mind2Web 2 Overview" width="800"/>
  <p><em>Overview of Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge</em></p>
</div>

## 🔗 Links

- [🏠 Homepage](https://osu-nlp-group.github.io/Mind2Web-2)
- [📖 Paper](https://arxiv.org/abs/2506.21506)
- [😊 Dataset (Coming Soon)](https://github.com/OSU-NLP-Group/Mind2Web-2/)

## 🆕 Updates

- **2025/06/26**: The GitHub repo is live. The manuscript is now on arXiv and accepted to ICML'25 Workshop on Computer Use Agents.

## ⚙️ Environment Setup

Run the following commands to set up the Python environment:

```bash
# Create and activate conda environment
conda create -n mind2web2 python=3.11
conda activate mind2web2

# Install the package in development mode
pip install -e .

# Install browsers for Playwright
playwright install
```

## 📁 Repo Structure

```
Mind2Web2-polish/
├── dataset/                 # Evaluation data and results
├── mind2web2/              # Main package
│   ├── api_tools/          # External API tools
│   ├── llm_client/         # LLM client implementations
│   ├── utils/              # Utility functions
│   ├── eval_runner.py      # Evaluation execution
│   ├── eval_toolkit.py     # Evaluation toolkit and utilities
│   ├── evaluator.py        # Core evaluation logic
│   └── verification_tree.py # Rubric tree implementation
├── pyproject.toml          # Package configuration
└── README.md              # This file
```

## 🚀 Run Evaluation

### 1. Prepare Your Data

Put your answer folder under the `dataset` directory. The folder should contain your agent's responses in markdown format.

### 2. Set up API Keys

Configure the necessary API keys for evaluation:

```bash
# Set up environment variables for OpenAI API
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# (Optional) Environment variables for Azure OpenAI
export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_ENDPOINT_URL="YOUR_AZURE_OPENAI_ENDPOINT_URL"
export AZURE_OPENAI_API_VERSION="2025-03-01-preview"

# Tool APIs for certain evaluation tasks
export GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```

### 3. Precache Webpages

Before running evaluation, you may want to precache the webpages to reduce evaluation latency. Loading webpages on-the-fly is very inefficient. 

Use the following script to precache:

```bash
# Coming Soon!
```

We also provide this lightweight script to fix the errors in the precached webpages (for example, some pages may be blocked by human verification):

```bash
# Coming Soon!
```

### 4. Run Evaluation

Execute the evaluation process:

```bash
# Coming Soon!
```


## 📝 Citation

If you find this work useful, please consider starring our repo and citing our papers:

```bibtex
@misc{gou2025mind2web2,
    title = {Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge}, 
    author = {Boyu Gou and Zanming Huang and Yuting Ning and Yu Gu and Michael Lin and Weijian Qi and Andrei Kopanev and Botao Yu and Bernal Jiménez Gutiérrez and Yiheng Shu and Chan Hee Song and Jiaman Wu and Shijie Chen and Hanane Nour Moussa and Tianshu Zhang and Jian Xie and Yifei Li and Tianci Xue and Zeyi Liao and Kai Zhang and Boyuan Zheng and Zhaowei Cai and Viktor Rozgic and Morteza Ziyadi and Huan Sun and Yu Su},
    year = {2025},
    eprint = {2506.21506},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI}
}
```
