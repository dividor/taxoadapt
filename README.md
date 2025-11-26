# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora
<br>Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han</a>


Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper: [TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora](https://arxiv.org/abs/2506.10737).

![Framework Diagram of TaxoAdapt](https://github.com/pkargupta/taxoadapt/blob/main/framework.png)

TaxoAdapt is a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs.

## Contents
  - [Setup](#setup)
    - [Virtual Environment](#virtual-environment)
    - [LLM Provider Configuration](#llm-provider-configuration)
    - [Arguments](#arguments)
  - [Custom Dataset](#custom-dataset)
  - [Video](#video)
  - [📖 Citation](#-citation)

## Setup

### Requirements
- Python 3.11
- GPU recommended (for vLLM local models)

### Virtual Environment
We recommend using a Python virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**If installation fails** (especially on macOS with vLLM), try the minimal requirements:
```bash
pip install -r requirements_api_only.txt
```
This installs only the packages needed for API-based LLM usage (excludes vLLM for local models).

### LLM Provider Configuration

**All LLM settings for '--llm api' are controlled via the `.env` file** - no code changes needed! This repository supports multiple LLM providers: **OpenAI**, **Azure OpenAI**, **Anthropic Claude**, and **Hugging Face**.

#### Quick Setup

1. **Copy the environment template:**
```bash
cp .env.example .env
```

2. **Edit `.env` and configure your LLM provider:**
```bash
# Open .env in your editor
nano .env  # or vim, code, etc.
```

3. **Set your provider and API key in `.env`:**
```bash
# Example: OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-xxxxx...
OPENAI_MODEL=gpt-4o-mini

# Example: Claude
# LLM_PROVIDER=claude
# ANTHROPIC_API_KEY=sk-ant-xxxxx...
# CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

**The `.env` file controls everything** - provider, model, and API credentials!

**Important:** `LLM_PROVIDER` must be set - there is no default. You control exactly which LLM is used.

4. **Test and run:**
```bash
python main.py                  # Run TaxoAdapt
```

See below for instructions on using your own data.

#### Get API Keys
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Azure OpenAI**: [portal.azure.com](https://portal.azure.com)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com/)
- **Hugging Face**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

#### Provider Details

<details>
<summary><b>OpenAI Configuration</b></summary>

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o, gpt-4
```
</details>

<details>
<summary><b>Azure OpenAI Configuration</b></summary>

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2023-05-15
```
</details>

<details>
<summary><b>Claude (Anthropic) Configuration</b></summary>

```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-3-5-sonnet-20241022  # or claude-3-opus, claude-3-haiku
```
</details>

<details>
<summary><b>Hugging Face Configuration</b></summary>

```bash
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_...
HUGGINGFACE_MODEL=meta-llama/Llama-3.3-70B-Instruct  # or Qwen/Qwen2.5-72B-Instruct
```
</details>

#### Switching Providers or Models

**Everything is controlled by the `.env` file!** To switch:

1. **Open `.env`:** `nano .env` (or your preferred editor)
2. **Change provider:** Edit `LLM_PROVIDER`
3. **Change model:** Edit the model variable (e.g., `OPENAI_MODEL`, `CLAUDE_MODEL`)
4. **Save and restart:** Re-run your script

**Examples of changes in `.env`:**
```bash
# Switch to GPT-4 (just change the model line)
OPENAI_MODEL=gpt-4

# Switch to Claude (change LLM_PROVIDER and add Claude config)
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-xxxxx...
CLAUDE_MODEL=claude-3-opus-20240229

# Switch to Hugging Face (change LLM_PROVIDER and add HF config)
LLM_PROVIDER=huggingface
HUGGINGFACE_API_KEY=hf_xxxxx...
HUGGINGFACE_MODEL=Qwen/Qwen2.5-72B-Instruct
```

## Running TaxoAdapt

### Arguments
The following are the primary arguments for TaxoAdapt (defined in main.py; modify as needed):

- `topic` $\rightarrow$ this is the topic of the corpus, e.g., "natural language processing", "robotics", etc.
- `dataset` $\rightarrow$ this is the name of the dataset, e.g., "llm_graph", "icra_2020", etc. The huggingface dataset should be added to the `construct_dataset` function in `main.py` (see below).
- `llm` $\rightarrow$ this is the LLM backend to use: "api" (API-based) or "vllm" (local). When using "api":
  - **All configuration comes from `.env` file**
  - **Must set `LLM_PROVIDER`** (openai, azure, claude, or huggingface) - no default
  - Set the model variable (e.g., `OPENAI_MODEL=gpt-4o`, `CLAUDE_MODEL=claude-3-5-sonnet-20241022`)
  - **The code reads from `.env` automatically** - just edit the file and restart
  - **You control exactly which LLM is used** - the system will fail if `LLM_PROVIDER` is not set
- `max_depth` $\rightarrow$ this is the maximum depth of each taxonomy to be constructed.
- `init_levels` $\rightarrow$ this is the number of initial levels to be constructed in the initial taxonomy.
- `max_density` $\rightarrow$ this is the maximum density of papers to be mapped to a node (or unmapped papers at a parent node) in the taxonomies. If a leaf node has more than `max_density` papers, it will trigger depth expansion at that node. If a parent node has more than `max_density` papers that are unmapped to any of its children, it will trigger width expansion at that node.

In `main.py`, we define the different dimensions of research for a specific topic, each of which will be constructed as a separate taxonomy. You can modify the dimensions in the `args.dimensions` list.

### Using Excel Files as Dataset

Instead of using predefined datasets, you can provide your own Excel file with custom data and specify your own taxonomy dimensions:

```bash
python main.py \
  --topic "Humanitarian evaluation" \
  --llm api \
  --dataset_sheet "./datasets/pdf_metadata_results_2023_2025_sampled.xlsx" \
  --dataset_sheet_tabname "PDF Metadata" \
  --dataset_title_fieldname "Title" \
  --dataset_abstract_fieldname "Abstractive Summary (map reduced)" \
  --dimensions_file "dimensions_definitions.txt"
```

**Excel requirements:**
- File must have headers in the first row
- Specify the exact column names for title and abstract fields
- Empty or NaN values will be skipped automatically

**Dimensions File (`--dimensions_file`):**
- **Required**: Path to a text file defining your taxonomy dimensions
- Default: `dimensions_definitions.txt` (includes 5 default NLP dimensions and 12 humanitarian evaluation examples)
- All dimensions listed in this file will be used to create separate taxonomies
- Format: `dimension_name|||Full definition text` (one per line, comments start with `#`)

**Defining Custom Dimensions:**

The dimensions file (`dimensions_definitions.txt`) defines all available dimensions for your analysis. Each dimension should have a detailed definition explaining what it represents:

**File Format:**
```
# This is a comment - lines starting with # are ignored
# Format: dimension_name|||definition text (use three pipe characters as separator)

Sector|||Sector: the specific humanitarian sector or domain that the research addresses, such as health, nutrition, shelter, education, protection, water and sanitation, or food security.
Modality|||Modality: the type or form of humanitarian intervention being evaluated, such as cash transfers, in-kind assistance, capacity building, service delivery, advocacy, or protection services.
```

**Creating Your Own Dimensions File:**

1. Copy `dimensions_definitions.txt` to a new file (e.g., `humanitarian_dimensions.txt`)
2. Keep only the dimensions you need or add new ones
3. Each line defines one dimension: `dimension_name|||definition text`
4. Use detailed definitions that will guide the LLM in building taxonomies
5. Point to your file with `--dimensions_file humanitarian_dimensions.txt`

**Important Notes:**
- The separator is three pipe characters: `|||`
- Definition text can be as long as needed (one line per dimension)
- All dimensions in the file will be used (no need to specify them separately)
- Lines starting with `#` are comments and will be ignored
- The file must contain at least one valid dimension
- The repository includes a default file with 5 NLP dimensions and 12 humanitarian evaluation dimensions as examples

**Default vs Custom Dimensions:**
- **Default dimensions** (tasks, datasets, methodologies, evaluation_methods, real_world_domains): Uses paper type classification before taxonomy construction
- **Custom dimensions**: All papers are classified through each taxonomy (no pre-filtering by paper type)

**Example Excel structure:**
| Title | Abstractive Summary (map reduced) |
|-------|-----------------------------------|
| Paper 1 Title | This evaluation examines... |
| Paper 2 Title | We assess the impact of... |

## Output Files

After running the taxonomy construction, the system generates several output files:

### Text and JSON Files
- `initial_taxo_{dimension}.txt` - Initial taxonomy structure for each dimension (text format)
- `final_taxo_{dimension}.txt` - Final taxonomy structure after expansion (text format)
- `final_taxo_{dimension}.json` - Final taxonomy in JSON format with full metadata

### Excel Output (`taxonomy_output.xlsx`)

The system automatically creates an Excel file with two tabs:

**Tab 1: Taxonomy**
- Column 1: `Dimension` - The dimension name
- Column 2: `Top Level` - Root level category (topic name)
- Column 3: `Second Level` - Child categories under the root
- Column 4: `Description` - Description of each category

**Tab 2: Examples**
- Column 1: `Dimension` - The dimension name
- Column 2: `Top Level` - Root level category
- Column 3: `Second Level` - Child category the paper was classified into
- Column 4: `Paper Title` - Title of the classified paper
- Column 5: `Abstract` - Abstract of the classified paper

This Excel file provides an easy-to-navigate view of your taxonomy structure and shows which papers were classified into each category.

### Error Logging (`errors.log`)

During taxonomy construction, JSON parsing errors and expansion failures are logged to `errors.log` instead of stopping execution. This allows the process to continue while capturing diagnostic information:

- **JSON Parse Errors**: Full raw output and cleaned output for debugging LLM responses
- **Expansion Failures**: Details about failed width/depth expansions after retry attempts
- **Empty Classifications**: Papers that couldn't be classified due to errors are marked as "unlabeled"

The system will:
- Print warnings to console for visibility
- Continue processing remaining papers
- Log complete error details to `errors.log` for later review
- Treat affected papers as unlabeled in the final taxonomy

**Review `errors.log` after completion** to identify any systematic issues with LLM responses that may need attention (e.g., improving prompts, adjusting API settings, or refining dimension definitions).

## Sampling Humanitarian Data

Before running the full taxonomy construction, you may want to create a balanced sample of your data. The `sample_humanitarian_data.py` script helps you create a stratified sample across agencies:

```bash
python sample_humanitarian_data.py \
  --dataset_sheet "./datasets/pdf_metadata_results_2023_2025.xlsx" \
  --dataset_sheet_tabname "PDF Metadata" \
  --dataset_title_fieldname "Title" \
  --dataset_abstract_fieldname "Abstractive Summary (map reduced)" \
  --agency_fieldname "Agency" \
  --sample_size 100 \
  --min_reports 10 \
  --random_seed 42
```

**Features:**
- **Stratified sampling**: Samples proportionally from each agency based on their representation
- **Minimum threshold**: Only includes agencies with at least `--min_reports` (default: 10)
- **Reproducible**: Uses `--random_seed` for consistent results
- **Balanced**: Ensures all eligible agencies are represented in the sample
- **Automatic output**: Creates `<input>_sampled.xlsx` in the same directory

**Example output:**
```
Agencies with >= 10 reports: 10
Total eligible records: 287
Final sample size: 50

Sample distribution by agency:
  WHO: 8/47 (17.0%)
  IOM: 8/43 (18.6%)
  UNHCR: 7/43 (16.3%)
  ...
```

After sampling, use the sampled file with `main.py` for taxonomy construction.

## Custom Dataset
To use a custom dataset, you need to add it to the `construct_dataset` function in `main.py`. You may add it as follows:

```python
elif args.dataset == 'dataset_name':
        ds = load_dataset("huggingface_dataset_name")
```
We assume that the dataset has a `title` and `abstract` field for each paper. If not, you can modify the function to extract the relevant fields from your dataset.

## Video
You can find a video explanation of the TaxoAdapt framework and its results on YouTube: [TaxoAdapt Video](https://youtu.be/dKUeSm9GoyU).


## 📖 Citation
Please cite the paper and star this repo if you use TaxoAdapt and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@article{kargupta2025taxoadapt,
  title={TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora},
  author={Kargupta, Priyanka and Zhang, Nan and Zhang, Yunyi and Zhang, Rui and Mitra, Prasenjit and Han, Jiawei},
  journal={arXiv preprint arXiv:2506.10737},
  year={2025}
}
```