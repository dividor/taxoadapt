import os
import json
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

from model_definitions import initializeLLM, promptLLM, constructPrompt
from prompts import multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema
from taxonomy import Node, DAG
from datasets import load_dataset
from expansion import expandNodeWidth, expandNodeDepth
from paper import Paper
from utils import clean_json_string

def construct_dataset_from_excel(args):
    """Construct dataset from Excel file with custom tab and field names"""
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    print(f"Reading Excel file: {args.dataset_sheet}")
    print(f"Tab: {args.dataset_sheet_tabname}")
    print(f"Title field: {args.dataset_title_fieldname}")
    print(f"Abstract field: {args.dataset_abstract_fieldname}")
    
    # Read Excel file
    df = pd.read_excel(args.dataset_sheet, sheet_name=args.dataset_sheet_tabname)
    
    internal_collection = {}
    
    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as f:
        internal_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            # Get title and abstract from specified field names
            title = str(row[args.dataset_title_fieldname]) if args.dataset_title_fieldname in row else ""
            abstract = str(row[args.dataset_abstract_fieldname]) if args.dataset_abstract_fieldname in row else ""
            
            # Skip if both are empty or NaN
            if (not title or title == 'nan') and (not abstract or abstract == 'nan'):
                continue
            
            temp_dict = {"Title": title, "Abstract": abstract}
            formatted_dict = json.dumps(temp_dict)
            f.write(f'{formatted_dict}\n')
            
            internal_collection[internal_count] = Paper(
                internal_count, 
                title, 
                abstract, 
                label_opts=args.dimensions, 
                internal=True
            )
            internal_count += 1
        
        print(f"Total # of Papers: {internal_count}")
    
    return internal_collection, internal_count


def construct_dataset(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    split = 'train'
    
    if args.dataset == 'emnlp_2024':
        ds = load_dataset("EMNLP/EMNLP2024-papers")
    elif args.dataset == 'emnlp_2022':
        ds = load_dataset("TimSchopf/nlp_taxonomy_data")
        split = 'test'
    elif args.dataset == 'cvpr_2024':
        ds = load_dataset("DeepNLP/CVPR-2024-Accepted-Papers")
    elif args.dataset == 'cvpr_2020':
        ds = load_dataset("DeepNLP/CVPR-2020-Accepted-Papers")
    elif args.dataset == 'iclr_2024':
        ds = load_dataset("DeepNLP/ICLR-2024-Accepted-Papers")
    elif args.dataset == 'iclr_2021':
        ds = load_dataset("DeepNLP/ICLR-2021-Accepted-Papers")
    elif args.dataset == 'icra_2024':
        ds = load_dataset("DeepNLP/ICRA-2024-Accepted-Papers")
    else:
        ds = load_dataset("DeepNLP/ICRA-2020-Accepted-Papers")
    
    
    internal_collection = {}

    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
        internal_count = 0
        id = 0
        for p in tqdm(ds[split]):
            if ('title' not in p) and ('abstract' not in p):
                continue
            
            temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
            formatted_dict = json.dumps(temp_dict)
            i.write(f'{formatted_dict}\n')
            internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=args.dimensions, internal=True)
            internal_count += 1
            id += 1
        print("Total # of Papers: ", internal_count)
    
    return internal_collection, internal_count

def initialize_DAG(args):
    ## we want to make this a directed acyclic graph (DAG) so maintain a list of the nodes
    roots = {}
    id2node = {}
    label2node = {}
    idx = 0

    for dim in args.dimensions:
        mod_topic = args.topic.replace(' ', '_').lower()
        mod_full_topic = args.topic.replace(' ', '_').lower() + f"_{dim}"
        root = Node(
                id=idx,
                label=mod_topic,
                dimension=dim
            )
        roots[dim] = root
        id2node[idx] = root
        label2node[mod_full_topic] = root
        idx += 1

    queue = deque([node for id, node in id2node.items()])

    # if taking long, you can probably parallelize this between the different taxonomies (expand by level)
    while queue:
        curr_node = queue.popleft()
        label = curr_node.label
        dim = curr_node.dimension
        # expand
        system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
        outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.01, top_p=1.0)[0]
        outputs = json.loads(clean_json_string(outputs)) if "```" in outputs else json.loads(outputs.strip())
        outputs = outputs['root_topic'] if 'root_topic' in outputs else outputs[label]

        # add all children
        for key, value in outputs.items():
            mod_key = key.replace(' ', '_').lower()
            mod_full_key = mod_key + f"_{dim}"
            if mod_full_key not in label2node:
                child_node = Node(
                        id=len(id2node),
                        label=mod_key,
                        dimension=dim,
                        description=value['description'],
                        parents=[curr_node]
                    )
                curr_node.add_child(mod_key, child_node)
                id2node[child_node.id] = child_node
                label2node[mod_full_key] = child_node
                if child_node.level < args.init_levels:
                    queue.append(child_node)
            elif label2node[mod_full_key] in label2node[label + f"_{dim}"].get_ancestors():
                continue
            else:
                child_node = label2node[mod_full_key]
                curr_node.add_child(mod_key, child_node)
                child_node.add_parent(curr_node)

    return roots, id2node, label2node


def main(args):

    print("######## STEP 1: LOAD IN DATASET ########")

    # Check if using Excel sheet or predefined dataset
    if args.dataset_sheet:
        # Validate that all Excel-related arguments are provided
        if not all([args.dataset_sheet_tabname, args.dataset_title_fieldname, args.dataset_abstract_fieldname]):
            raise ValueError(
                "When using --dataset_sheet, you must also provide:\n"
                "  --dataset_sheet_tabname\n"
                "  --dataset_title_fieldname\n"
                "  --dataset_abstract_fieldname"
            )
        internal_collection, internal_count = construct_dataset_from_excel(args)
    else:
        internal_collection, internal_count = construct_dataset(args)
    
    print(f'Internal: {internal_count}')

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    roots, id2node, label2node = initialize_DAG(args)

    for dim in args.dimensions:
        with open(f'{args.data_dir}/initial_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                roots[dim].display(0, indent_multiplier=5)

    print("######## STEP 3: CLASSIFY PAPERS BY DIMENSION (TASK, METHOD, DATASET, EVAL, APPLICATION, etc.) ########")

    dags = {dim:DAG(root=root, dim=dim) for dim, root in roots.items()}

    # Check if we're using default NLP dimensions or custom dimensions
    default_dimensions = {"tasks", "methodologies", "datasets", "evaluation_methods", "real_world_domains"}
    using_default_dims = set(args.dimensions) == default_dimensions
    
    if using_default_dims:
        # For default dimensions, use the type classification schema (original algorithm)
        prompts = [constructPrompt(args, type_cls_system_instruction, type_cls_main_prompt(paper)) for paper in internal_collection.values()]
        outputs = promptLLM(args=args, prompts=prompts, schema=TypeClsSchema, max_new_tokens=500, json_mode=True, temperature=0.1, top_p=0.99)
        outputs = [json.loads(clean_json_string(c)) for c in outputs]

        for r in roots:
            roots[r].papers = {}
        type_dist = {dim:[] for dim in args.dimensions}
        for p_id, out in enumerate(outputs):
            internal_collection[p_id].labels = {}
            for key, val in out.items():
                if val:
                    type_dist[key].append(internal_collection[p_id])
                    internal_collection[p_id].labels[key] = []
                    roots[key].papers[p_id] = internal_collection[p_id]
        
        print(str({k:len(v) for k,v in type_dist.items()}))
    else:
        # For custom dimensions, assign all papers to all dimensions
        # (no pre-classification by paper type, all papers go through all taxonomies)
        print("Using custom dimensions - all papers will be classified through each taxonomy")
        for r in roots:
            roots[r].papers = {}
        type_dist = {dim:[] for dim in args.dimensions}
        
        for p_id, paper in enumerate(internal_collection.values()):
            paper.labels = {}
            for dim in args.dimensions:
                type_dist[dim].append(paper)
                paper.labels[dim] = []
                roots[dim].papers[p_id] = paper
        
        print(str({k:len(v) for k,v in type_dist.items()}))


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    visited = set()
    queue = deque([roots[r] for r in roots])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    for dim in args.dimensions:
        with open(f'{args.data_dir}/final_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)

        with open(f'{args.data_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='natural language processing',
                       help='Topic of the corpus (e.g., "natural language processing", "robotics")')
    parser.add_argument('--dataset', type=str, default='llm_graph',
                       help='Predefined dataset name (e.g., "emnlp_2024", "cvpr_2024")')
    parser.add_argument('--llm', type=str, default='api',
                       help='LLM backend: "api" (API-based, configured via .env) or "vllm" (local)')
    parser.add_argument('--max_depth', type=int, default=2,
                       help='Maximum depth of each taxonomy')
    parser.add_argument('--init_levels', type=int, default=1,
                       help='Number of initial levels in the taxonomy')
    parser.add_argument('--max_density', type=int, default=40,
                       help='Maximum density of papers per node')
    parser.add_argument('--dimensions_file', type=str, default='dimensions_definitions.txt',
                       help='Path to dimensions definitions file (default: dimensions_definitions.txt)')
    
    # Excel sheet arguments (alternative to --dataset)
    parser.add_argument('--dataset_sheet', type=str, default=None,
                       help='Path to Excel file (alternative to --dataset)')
    parser.add_argument('--dataset_sheet_tabname', type=str, default=None,
                       help='Excel sheet/tab name to read from')
    parser.add_argument('--dataset_title_fieldname', type=str, default=None,
                       help='Column name for paper titles')
    parser.add_argument('--dataset_abstract_fieldname', type=str, default=None,
                       help='Column name for paper abstracts')
    
    args = parser.parse_args()

    # Load dimensions from dimensions file
    if not os.path.exists(args.dimensions_file):
        raise ValueError(f"Dimensions file not found: {args.dimensions_file}")
    
    dimensions = []
    with open(args.dimensions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '|||' in line:
                parts = line.split('|||', 1)
                if len(parts) == 2:
                    dim_name = parts[0].strip()
                    dimensions.append(dim_name)
    
    if not dimensions:
        raise ValueError(f"No dimensions found in {args.dimensions_file}. Please add at least one dimension.")
    
    args.dimensions = dimensions
    print(f"Loaded {len(dimensions)} dimensions from {args.dimensions_file}: {', '.join(dimensions)}")

    # Set data directory based on dataset source
    if args.dataset_sheet:
        # Use Excel filename (without extension) as data directory name
        excel_basename = os.path.splitext(os.path.basename(args.dataset_sheet))[0]
        args.data_dir = f"datasets/{excel_basename.lower().replace(' ', '_')}"
        args.internal = f"{excel_basename}.txt"
    else:
        args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}"
        args.internal = f"{args.dataset}.txt"

    main(args)