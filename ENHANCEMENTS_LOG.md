# Enhancements Log

This document lists all features and improvements added to this fork compared to the original source repository.

---

## Multi-Provider LLM Support

### Overview
Extended the codebase to support multiple LLM providers beyond the original OpenAI implementation.

### Features
- **Unified LLM Provider Interface**: Abstract `LLMProvider` class with concrete implementations for:
  - OpenAI (original)
  - Azure OpenAI
  - Claude (Anthropic)
  - Hugging Face (via OpenAI-compatible Router API)
- **Provider Factory Pattern**: `LLMProviderFactory` for dynamic provider instantiation
- **Environment-based Configuration**: All LLM settings controlled via `.env` file
- **Strict Provider Control**: `LLM_PROVIDER` environment variable (required, no fallback) ensures explicit user control

### Implementation
- Created `api/llm_provider.py` with provider abstraction
- Updated `model_definitions.py` to use unified provider interface
- Modified `classification.py` to support both API and vLLM prompt formats
- Changed `--llm` flag options from `'gpt'` to `'api'` and `'vllm'`

### Files Added
- `api/llm_provider.py`
- `.env.example`
- `.gitignore`

### Files Modified
- `model_definitions.py`
- `classification.py`
- `main.py`

---

## Excel Dataset Input Support

### Overview
Added ability to use custom Excel files as dataset input, providing flexibility beyond predefined datasets.

### Features
- **Excel File Loading**: Read datasets directly from Excel spreadsheets
- **Flexible Column Mapping**: Specify custom column names for title and abstract fields
- **Tab Selection**: Support for multi-tab Excel files with tab name specification
- **Automatic Directory Creation**: Derives output directory from Excel filename

### Implementation
- Added `construct_dataset_from_excel()` function in `main.py`
- New command-line arguments:
  - `--dataset_sheet`: Path to Excel file
  - `--dataset_sheet_tabname`: Sheet/tab name to read
  - `--dataset_title_fieldname`: Column name for paper titles
  - `--dataset_abstract_fieldname`: Column name for paper abstracts

### Files Modified
- `main.py`

### Dependencies Added
- `pandas>=2.0.3`
- `openpyxl>=3.1.5`

---

## Custom Dimension Definitions

### Overview
Implemented flexible taxonomy dimension system allowing users to define custom dimensions beyond the default NLP dimensions.

### Features
- **External Dimension Definitions**: Load dimension definitions from text file
- **Custom Dimension Support**: Define domain-specific dimensions (e.g., humanitarian evaluation)
- **Validation System**: Ensures all dimensions used are explicitly defined
- **Single Source of Truth**: One file (`dimensions_definitions.txt`) for all dimension definitions
- **Conditional Classification Logic**: Adapts paper classification based on default vs. custom dimensions

### Implementation
- Created `dimensions_definitions.txt` with default NLP and example humanitarian dimensions
- Added `--dimensions_file` argument to specify dimension definition file
- Updated `prompts.py` to load and validate dimensions from file
- Modified `main.py` to skip `TypeClsSchema` classification for custom dimensions
- Added validation in `prompts.py` and `ablations/no_cluster_expansion.py`

### Files Added
- `dimensions_definitions.txt`

### Files Modified
- `main.py`
- `prompts.py`
- `ablations/no_cluster_expansion.py`

---

## Robust JSON Parsing

### Overview
Enhanced JSON parsing to handle various LLM output formats and edge cases.

### Features
- **Markdown Block Handling**: Strips markdown code blocks (````json` wrappers)
- **Extra Text Removal**: Removes explanatory text after JSON content
- **Python-to-JSON Conversion**: Converts Python booleans (`True`/`False`/`None`) to JSON format (`true`/`false`/`null`)
- **Unquoted Property Names**: Fixes JavaScript-style unquoted property names
- **Empty Output Handling**: Graceful handling of empty or malformed LLM responses

### Implementation
- Enhanced `clean_json_string()` function in `utils.py`
- Applied consistent JSON cleaning across all parsing points:
  - `main.py` (paper classification)
  - `taxonomy.py` (node classification)
- Added error handling for `JSONDecodeError` with empty list fallback

### Files Modified
- `utils.py`
- `main.py`
- `taxonomy.py`

---

## Excel Output Generation

### Overview
Added structured Excel output with taxonomy hierarchy and paper examples for better data visualization.

### Features
- **Multi-tab Excel Output**: Generates `taxonomy_output.xlsx` with two tabs:
  1. **Taxonomy Tab**: Complete taxonomy structure
     - Columns: Dimension, Top Level, Second Level, Description
  2. **Examples Tab**: Paper mappings to taxonomy nodes
     - Columns: Dimension, Top Level, Second Level, Paper Title, Abstract
- **Automatic Generation**: Created at end of taxonomy construction process
- **Progress Reporting**: Prints row counts for each tab

### Implementation
- Added Excel generation code block in `main.py` (STEP 6)
- Uses `pandas.ExcelWriter` with `openpyxl` engine
- Iterates through taxonomy structure to populate both tabs

### Files Modified
- `main.py`

---

## Data Sampling Utility

### Overview
Created standalone script for balanced stratified sampling of humanitarian datasets.

### Features
- **Stratified Sampling**: Balanced sampling across agencies
- **Minimum Threshold**: Filter agencies with fewer than specified minimum reports
- **Proportional Distribution**: Maintains relative proportions of eligible agencies
- **Reproducible Results**: Random seed support for consistent sampling
- **Sample Size Adjustment**: Automatic distribution adjustment to meet target sample size
- **Detailed Reporting**: Comprehensive output showing sampling distribution

### Implementation
- Created `sample_humanitarian_data.py` with command-line interface
- Reuses Excel input flags from `main.py` for consistency
- Additional parameters:
  - `--agency_fieldname`: Column containing agency information
  - `--sample_size`: Target number of records to sample
  - `--min_reports`: Minimum reports per agency threshold (default: 10)
  - `--random_seed`: Seed for reproducibility (default: 42)
  - `--output_path`: Output file path

### Files Added
- `sample_humanitarian_data.py`

---

## Dependency Management

### Overview
Improved dependency management for stability and compatibility.

### Features
- **Version Pinning**: All dependencies pinned to exact working versions
- **API-Only Installation**: Lighter installation option without vLLM
- **Python Version Specification**: Supports Python 3.8-3.11
- **Conflict Resolution**: Resolved numpy/mistral-common version conflicts

### Implementation
- Pinned all versions in `requirements.txt`
- Created `requirements_api_only.txt` for API-only installations
- Updated numpy to `>=1.25.0,<2.0.0` for compatibility

### Files Modified
- `requirements.txt`
- `requirements_api_only.txt` (created)

---

## Documentation Improvements

### Overview
Comprehensive documentation updates for all new features.

### Features
- **Virtual Environment Setup**: Step-by-step venv instructions
- **LLM Provider Configuration**: Detailed setup for each provider
- **Excel Input Guide**: Complete guide with examples
- **Dimension System Documentation**: Explanation of custom dimensions
- **Output Files Reference**: Documentation of all output formats
- **Troubleshooting Section**: Common issues and solutions
- **Sampling Script Guide**: Usage instructions for data sampling

### Files Modified
- `README.md`

---

## Git and Configuration Management

### Overview
Improved repository configuration and git management.

### Features
- **Environment Variable Template**: `.env.example` for easy configuration
- **Enhanced .gitignore**: 
  - Excludes `.env` files
  - Excludes test data directories (`datasets/test_data_*/`)
  - Excludes Excel temp files (`~$*.xlsx`, `~$*.xls`)
  - Python cache and virtual environments
- **Test Data Exclusion**: Removed test data from git tracking

### Files Added
- `.env.example`

### Files Modified
- `.gitignore`

---

## Summary Statistics

### New Files Created
- `api/llm_provider.py`
- `.env.example`
- `.gitignore`
- `dimensions_definitions.txt`
- `sample_humanitarian_data.py`
- `requirements_api_only.txt`
- `ENHANCEMENTS_LOG.md`

### Files Modified
- `main.py`
- `model_definitions.py`
- `classification.py`
- `prompts.py`
- `taxonomy.py`
- `utils.py`
- `ablations/no_cluster_expansion.py`
- `requirements.txt`
- `README.md`

### New Dependencies
- `anthropic==0.39.0`
- `huggingface_hub==0.26.2`
- `python-dotenv==1.0.1`
- `pandas==2.0.3`
- `openpyxl==3.1.5`

### New Command-Line Arguments
- `--llm`: Changed from `'gpt'` to `'api'` option
- `--dataset_sheet`: Excel file path
- `--dataset_sheet_tabname`: Excel sheet name
- `--dataset_title_fieldname`: Title column name
- `--dataset_abstract_fieldname`: Abstract column name
- `--dimensions_file`: Dimension definitions file path

---

## Robust Error Handling and LLM Provider Consistency

### Overview
Implemented strict error handling for JSON parsing failures and fixed LLM provider switching issues to ensure API mode works consistently throughout execution.

### Features
- **Hard Stop on JSON Parse Errors**: All JSON parsing failures now raise exceptions with full error details instead of silently continuing with empty data
- **Complete Error Output**: Failed JSON responses are printed in full (no truncation) to aid debugging
- **LLM Provider Preservation**: Fixed issue where `expansion.py` was hardcoding LLM provider switches, now preserves user's `--llm` choice
- **Detailed Error Messages**: Clear, formatted error messages with separators for easy identification

### Implementation Details

#### JSON Parse Error Handling
- **taxonomy.py**: Modified `classify_node()` to raise `RuntimeError` on JSON parse errors
  - Prints full raw output and cleaned output
  - No longer returns empty classifications on parse failures
  
#### Expansion Error Handling  
- **expansion.py**: Updated both `expandNodeWidth()` and `expandNodeDepth()` functions
  - Raise `RuntimeError` after 5 failed parse attempts (previously returned empty results)
  - Print full problematic JSON output for debugging
  
#### LLM Provider Consistency
- **expansion.py**: Fixed hardcoded `args.llm = 'vllm'` switches
  - Now saves and restores original LLM provider setting
  - Ensures `--llm api` works throughout entire execution
  - Prevents `KeyError: 'vllm'` when running in API-only mode

#### Enhanced JSON Cleaning
- **utils.py**: Updated `clean_json_string()` to handle YAML-style separators
  - Removes `---` markers that some LLMs add around JSON output
  - Handles both wrapped (`---...---`) and single `---` markers
  - Prevents `JSONDecodeError` from improperly formatted LLM responses

### Error Message Format
```
================================================================================
CRITICAL ERROR: JSON parse error for prompt X: [error details]
================================================================================
Raw output (FULL):
[complete raw JSON response]
================================================================================
Cleaned output (FULL):
[complete cleaned JSON]
================================================================================
```

### Files Modified
- `taxonomy.py`
- `expansion.py`
- `utils.py`

### Impact
- **Improved Debugging**: Full JSON output on errors makes API issues easier to diagnose
- **Fail-Fast Behavior**: Stops execution immediately on parse errors instead of propagating bad data
- **API Mode Stability**: `--llm api` now works reliably without vLLM dependency or configuration

---

## Backward Compatibility

All enhancements maintain backward compatibility with the original codebase:
- Original vLLM functionality preserved with `--llm vllm`
- Default NLP dimensions still supported
- Original dataset loading mechanism still functional
- Core taxonomy construction algorithm unchanged

