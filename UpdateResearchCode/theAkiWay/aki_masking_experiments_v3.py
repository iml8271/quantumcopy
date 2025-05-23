# %% Imports
import torch
import numpy as np
import matplotlib
# Use Agg backend for non-interactive environments (e.g., servers, notebooks without GUI)
# Should be set before importing pyplot
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import networkx as nx # Added for graph visualization
from qiskit.quantum_info import DensityMatrix
from qiskit import QuantumCircuit # Added for type hinting if needed
import itertools
import collections
import pandas as pd # For displaying statistics
import time
import math
import os # Added for path handling
import re # Added for filename sanitization
from datetime import datetime # Added for timestamped filenames
import pprint

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Assuming genQC library is installed and accessible
try:
    from genQC.imports import *
    from genQC.pipeline.diffusion_pipeline import DiffusionPipeline
    from genQC.inference.infer_srv import convert_tensors_to_srvs, schmidt_rank_vector
    import genQC.platform.qcircuit_dataset_construction as data_const
    from genQC.platform.simulation.qcircuit_sim import instruction_name_to_qiskit_gate
    import genQC.util as util
    from einops import repeat
    from tqdm.auto import tqdm # For progress bars
    from beartype.typing import Optional, List # For type hinting
except ImportError as e:
    print(f"Error importing genQC library: {e}")
    print("Please ensure the genQC library and its dependencies are installed.")
    # Provide dummy functions or raise error if essential parts are missing
    # For demonstration, we'll let it fail if imports don't work.
    raise

# %% Configuration
DEVICE = util.infer_torch_device()  # use cuda if we can
util.MemoryCleaner.purge_mem()     # clean existing memory alloc
print(f"Using device: {DEVICE}")

# --- Control Flag ---
GENERATE_VISUALS_ONLY = False # Set to True to only generate mask/graph images and skip experiments
# --------------------

# --- Model Parameters (Only needed if GENERATE_VISUALS_ONLY is False) ---
MODEL_NAME = "Floki00/qc_srv_3to8qubit"
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 40
# Start denoising later in the process for pattern filling (empirical)
T_START_FRAC = 0.5 # Fraction of total timesteps to start denoising from for pattern generation

# --- Experiment Parameters ---
QUBIT_COUNTS = range(4, 9) # 4, 5, 6, 7, 8 qubits
SAMPLES_PER_EXPERIMENT = 4096 # Number of circuits to generate per config (if running experiments)
MAX_GATES = 36  # Max sequence length for the model / mask
GATE_SLOT_DURATION = 3 # How many time steps allocated per connection in the mask
CIRCUITS_TO_SAVE = 8 # Max number of circuit images to save per config (if running experiments)
RESULTS_OUTPUT_DIR = f"diffusion_analysis_results_{timestamp}" # Directory to save CSV results
IMAGE_OUTPUT_DIR = f"diffusion_analysis_circuit_images_{timestamp}" # Directory to save circuit images and masks

# %% Helper Functions

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters unsafe for filenames."""
    # Remove brackets and spaces
    sanitized = filename.replace("[", "").replace("]", "").replace(" ", "")
    # Replace commas and slashes (common in geometry names) with underscores
    sanitized = sanitized.replace(",", "_").replace("/", "_").replace("\\", "_")
    # Remove any remaining non-alphanumeric characters except underscores and hyphens
    sanitized = re.sub(r'[^\w\-]+', '', sanitized)
    # Avoid excessively long names
    return sanitized[:100] # Limit length

# %% Define Hardware Geometries (Deduplicated)

# Helper function to get a canonical representation for checking duplicates
def get_canonical_connectivity(connections):
    """
    Creates a canonical representation of connectivity for easy comparison.
    Sorts qubits within each tuple, then sorts the list of tuples.
    Returns a tuple of tuples for hashing.
    """
    if not connections:
        return tuple()
    # Ensure pairs are tuples and sort elements within each pair
    sorted_pairs = [tuple(sorted(pair)) for pair in connections]
    # Sort the list of pairs
    canonical_list = sorted(sorted_pairs)
    return tuple(canonical_list) # Convert to tuple for hashing

# Initial dictionary with potential duplicates and updated names
initial_geometries = {
    4: {
        "Linear_4Q_(Generic)": [(0, 1), (1, 2), (2, 3)],
        "Grid_2x2_(Superconducting)": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "IBM_HeavyHex_4Q_T": [(0, 1), (1, 2), (1, 3)],
        "Star_4Q_(Trapped_Ion)": [(0, 1), (0, 2), (0, 3)],
        "All-to-All_4Q_(Trapped_Ion)": [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
    },
    5: {
        "Linear_5Q_(Generic)": [(0, 1), (1, 2), (2, 3), (3, 4)],
        # "IBM_Quito_5Q": [(0,1), (1,2), (1,3), (3,4)], # Duplicate of Google_Sycamore_5Q_Cross
        "Square_Tail_5Q_(Superconducting)": [(0,1), (1,2), (2,3), (3,0), (0,4)],
        "Ring_5Q_(Superconducting)": [(0,1), (1,2), (2,3), (3,4), (4,0)],
        "Google_Sycamore_5Q_Cross": [(0,1), (1,2), (1,3), (3,4)],
        "All-to-All_5Q_(Trapped_Ion)": [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)],
    },
    6: {
        "Linear_6Q_(Generic)": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "Grid_2x3_(Superconducting)": [(0,1), (1,2), (3,4), (4,5), (0,3), (1,4), (2,5)],
        "Prism_Graph_CL3_(Trapped_Ion)": [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)],
        "IBM_HeavyHex_6Q_Fragment": [(0,1), (1,2), (1,3), (3,4), (4,5), (2,5)],
        "Ring_6Q_(Superconducting)": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)],
        "Rigetti_Aspen_6Q_Fragment": [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5), (1,4)],
        "All-to-All_6Q_(Trapped_Ion)": [(0,1), (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)],
    },
    7: {
        "Linear_7Q_(Generic)": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5,6)],
        "IBM_Perth_7Q": [(0,1), (1,2), (1,3), (3,5), (4,5), (5,6), (2,4)],
        "Grid_2x3_Tail_7Q_(Superconducting)": [(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5),(2,6)],
        "Ring_Tail_7Q_(Superconducting)": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,6)],
        "Rigetti_Aspen_7Q_Fragment": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (0,7), (1,6)],
        "IBM_HeavyHex_7Q_Alt": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1,4)],
        "All-to-All_7Q_(Trapped_Ion)": [
                            (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,2), (1,3), (1,4), (1,5),
                            (1,6), (2,3), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6), (4,5), (4,6),
                            (5,6)],
    },
    8: {
        "Linear_8Q_(Generic)": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        "Grid_2x4_(Superconducting)": [(0,1), (1,2), (2,3), (4,5), (5,6), (6,7), (0,4), (1,5), (2,6), (3,7)],
        "Prism_Graph_CL4_(Trapped_Ion)": [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)],
        "Two_Squares_Linked_8Q_(Superconducting)": [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4)],
        "IBM_HeavyHex_8Q_Tiled": [(0,1),(1,2),(1,3),(3,5),(4,5),(5,6),(2,4),(6,7)],
        "Rigetti_Aspen_8Q_Ring": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)],
        "Rigetti_Aspen_8Q_Internal": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0), (0,4), (1,5)],
        "IBM_HeavyHex_8Q_Fragment": [(0,1), (1,2), (2,3), (1,4), (4,5), (5,6), (6,7), (4,7)],
        "All-to-All_8Q_(Trapped_Ion)": [
                            (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (1,2), (1,3), (1,4),
                            (1,5), (1,6), (1,7), (2,3), (2,4), (2,5), (2,6), (2,7), (3,4), (3,5),
                            (3,6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7)],
    }
}

# --- Deduplicate Geometries ---
HARDWARE_GEOMETRIES = {} # Start fresh
print("Deduplicating geometries...")
for num_qubits, geometries in initial_geometries.items():
    HARDWARE_GEOMETRIES[num_qubits] = {}
    seen_canonicals = set()
    for name, connections in geometries.items():
        canonical = get_canonical_connectivity(connections)
        if canonical not in seen_canonicals:
            HARDWARE_GEOMETRIES[num_qubits][name] = connections
            seen_canonicals.add(canonical)
        else:
            print(f"Removed duplicate geometry: {num_qubits}Q - {name} (same as an earlier entry)")

# --- Add Baseline Geometry ---
for q_count in list(HARDWARE_GEOMETRIES.keys()): # Use list to avoid modifying dict during iteration
    if "Baseline_Unconstrained" not in HARDWARE_GEOMETRIES[q_count]:
        HARDWARE_GEOMETRIES[q_count]["Baseline_Unconstrained"] = [] # Empty list signifies unconstrained
        print(f"Added Baseline_Unconstrained for {q_count} qubits.")

# --- Final Result ---
print("\n--- Final HARDWARE_GEOMETRIES (Deduplicated and Updated Names) ---")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(HARDWARE_GEOMETRIES)


# %% Define SRV Generation (Modified)

def generate_srvs(num_qubits: int) -> List[List[int]]:
    """
    Generates all unique, physically meaningful Schmidt Rank Vectors
    for a given number of qubits. Excludes SRVs with exactly one '2'.
    """
    if num_qubits < 1:
        return []

    srvs = set()
    # Generate all combinations of 1s and 2s
    for combo in itertools.product([1, 2], repeat=num_qubits):
        # Sort the combo to make SRVs like [1, 2, 1] equivalent to [1, 1, 2]
        # Store as tuple for hashing in the set
        srvs.add(tuple(sorted(list(combo))))

    # Convert back to list of lists
    unique_srvs = [list(srv) for srv in srvs]

    # --- Filter out SRVs with exactly one '2' ---
    filtered_srvs = []
    for srv in unique_srvs:
        if srv.count(2) != 1:
            filtered_srvs.append(srv)
    # --------------------------------------------

    # Sort the final list for consistent order
    sorted_filtered_srvs = sorted(filtered_srvs)
    return sorted_filtered_srvs

# %% Mask Creation Function (Handles Baseline)

def create_mask_from_connectivity(num_qubits: int,
                                  geo_name: str, # Added geo_name argument
                                  connectivity: List[tuple],
                                  max_gates: int,
                                  gate_slot_duration: int) -> torch.Tensor:
    """
    Creates a mask tensor based on qubit connectivity or creates an unconstrained
    mask if geo_name is 'Baseline_Unconstrained'.

    Mask: 1s mean the model should generate/fill this location,
          0s mean the location is fixed (usually empty).
    """
    # --- Handle Baseline Case ---
    if geo_name == "Baseline_Unconstrained":
        mask = torch.ones((num_qubits, max_gates), device=DEVICE, dtype=torch.float32)
        effective_num_gates = max_gates # For baseline, all gates are potentially usable
        return mask, effective_num_gates
    # --- End Baseline Case ---

    # --- Standard Connectivity-Based Mask ---
    mask = torch.zeros((num_qubits, max_gates), device=DEVICE, dtype=torch.float32) # Use float for pipeline

    current_time_slot = 0
    max_interaction_time = 0 # Track the last time slot used by any interaction

    # Allocate sequential slots for 2-qubit interactions based on connectivity
    for q1, q2 in connectivity:
        start_idx = current_time_slot
        end_idx = start_idx + gate_slot_duration
        if end_idx > max_gates:
            # print(f"Warning: Ran out of gate slots ({max_gates}) for connectivity mask. Stopping allocation.")
            break # Stop adding connections if we exceed max_gates

        # Check bounds before assigning to mask
        if 0 <= q1 < num_qubits and 0 <= q2 < num_qubits:
             # Mark both qubits as active during this time slot
             mask[q1, start_idx:end_idx] = 1
             mask[q2, start_idx:end_idx] = 1
             # Only advance time slot if the connection was valid and added
             current_time_slot = end_idx
             # Update the maximum time any interaction occurred
             max_interaction_time = max(max_interaction_time, end_idx)
        # else:
             # print(f"Warning: Qubit indices {q1}, {q2} out of bounds for {num_qubits} qubits in connectivity list.")

    # Calculate effective number of gates the mask allows (last column index + 1)
    effective_num_gates = 0
    if mask.sum() > 0:
      # Find the last column index where any qubit has a '1'
      last_gate_idx = -1
      try:
          # Find indices where the sum across qubits is greater than 0
          active_cols = torch.where(mask.sum(dim=0) > 0)[0]
          if len(active_cols) > 0:
              last_gate_idx = active_cols.max().item()
              effective_num_gates = last_gate_idx + 1
      except IndexError: # Handle case where mask is all zeros despite sum > 0 (should not happen)
          effective_num_gates = 0
      except RuntimeError as e: # Catch potential errors on specific devices/versions
          print(f"Runtime error calculating effective gates: {e}. Setting to 0.")
          effective_num_gates = 0

    return mask, effective_num_gates

# %% Adapted Generation Function (Only needed if GENERATE_VISUALS_ONLY is False)

if not GENERATE_VISUALS_ONLY:
    @torch.no_grad() # Disable gradient calculations for inference
    def generate_circuits_with_mask(
        pipeline: DiffusionPipeline,
        prompt: str,
        mask: torch.Tensor,
        samples: int,
        system_size: int, # Usually == num_of_qubits
        num_of_qubits: int,
        max_gates: int,
        target_num_gates: Optional[int] = None,
        target_num_bits: Optional[int] = None,
        guidance_scale: float = 7.5,
        t_start_index: int = 0,
        no_bar: bool = True, # Set True to hide progress bar during experiment
        padd_tok_val: Optional[int] = None
    ):
        """
        Generates circuits using latent filling based on a mask and prompt.
        Returns lists of generated circuits and their SRVs.
        """
        # Use provided target_num_gates directly if available (esp. for baseline)
        # Otherwise, estimate from mask
        if not exists(target_num_gates) or target_num_gates <= 0:
            last_gate_idx = -1
            if mask.sum() > 0:
               try:
                   active_cols = torch.where(mask.sum(dim=0) > 0)[0]
                   if len(active_cols) > 0:
                       last_gate_idx = active_cols.max().item()
               except IndexError:
                    last_gate_idx = -1
               except RuntimeError:
                    last_gate_idx = -1
            calc_target_gates = last_gate_idx + 1
            if calc_target_gates <= 0 : calc_target_gates = max_gates // 2 # Default if mask is effectively empty
            target_num_gates = calc_target_gates # Use calculated value
            # print(f"Using estimated target_num_gates: {target_num_gates}")
        # Clamp target_num_gates to max_gates
        target_num_gates = min(target_num_gates, max_gates)


        if not exists(target_num_bits):
            target_num_bits = num_of_qubits

        if not exists(padd_tok_val):
             padd_tok_val = len(pipeline.gate_pool) + 1 # Default padding token index


        # --- Prepare initial state (empty circuit based on mask) ---
        org_image = torch.zeros((1, system_size, max_gates), device=DEVICE, dtype=torch.int32)

        # Apply padding based on target dimensions BEFORE embedding
        # Pad unused gate slots beyond the effective target length
        padd_pos = target_num_gates # Pad directly after the effective gates used by mask/baseline
        padd_pos = min(int(padd_pos), max_gates) # Ensure it doesn't exceed max_gates

        if padd_pos < max_gates:
            org_image[:, :, padd_pos:] = padd_tok_val
        # Pad unused qubit lines
        if target_num_bits < system_size:
            org_image[:, target_num_bits:, :] = padd_tok_val


        # --- Embed initial state ---
        emb_org_image = pipeline.model.embedd_clrs(org_image)
        emb_org_images = repeat(emb_org_image, '1 ... -> b ...', b=samples)


        # --- Prepare Condition (Prompt) ---
        c = pipeline.text_encoder.tokenize_and_push_to_device(str(prompt))
        c = repeat(c, '1 ... -> b ...', b=samples)
        uc = {} # Assuming no unconditional guidance needed


        # --- Latent Filling ---
        qubit_mask_float = mask.float() # Ensure it's float

        try:
            with torch.cuda.amp.autocast(enabled=(DEVICE != 'cpu')): # Use mixed precision if on GPU
                out_tensor = pipeline.latent_filling(
                    emb_org_images,
                    qubit_mask_float,
                    c=c,
                    uc=uc,
                    g=guidance_scale,
                    no_bar=no_bar,
                    t_start_index=t_start_index
                )
        except Exception as e:
            print(f"\nError during pipeline.latent_filling: {e}")
            # Clean up memory potentially allocated on GPU during failed attempt
            del emb_org_images, c, uc, qubit_mask_float
            util.MemoryCleaner.purge_mem()
            return [], [], 0 # Return empty lists and 0 errors on failure

        # --- Decode and Convert ---
        out_tensor = pipeline.model.invert_clr(out_tensor)
        # Trim to target size - important to use target_num_gates derived from mask/baseline!
        out_tensor = out_tensor[:, :num_of_qubits, :target_num_gates]

        unique_out_tensor = torch.unique(out_tensor, dim=0)

        # Move tensor to CPU for conversion - safer as qiskit might not handle GPU tensors
        unique_out_tensor_cpu = unique_out_tensor.cpu()
        del out_tensor, unique_out_tensor # Free GPU memory
        util.MemoryCleaner.purge_mem()


        # Convert potentially large number of tensors - might be slow
        qc_list, error_cnt, srv_list = convert_tensors_to_srvs(
            unique_out_tensor_cpu,
            pipeline.gate_pool,
            place_barrier=False, # Don't place barriers for SRV calculation
            # Consider adding num_workers if conversion is bottleneck
        )

        # Filter out circuits where SRV calculation failed (returned None)
        valid_qc_list = []
        valid_srv_list = []
        for qc, srv in zip(qc_list, srv_list):
            if srv is not None:
                 valid_qc_list.append(qc)
                 valid_srv_list.append(srv)

        return valid_qc_list, valid_srv_list, error_cnt

# %% Load Model (Only if needed)

pipeline = None
PADD_TOKEN = -1 # Default value
if not GENERATE_VISUALS_ONLY:
    print("Loading diffusion pipeline...")
    try:
        pipeline = DiffusionPipeline.from_pretrained(MODEL_NAME, DEVICE)
        pipeline.guidance_sample_mode = "rescaled" # As per example
        pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        print("Pipeline loaded.")

        # Calculate t_start_index based on fraction
        t_start_index = int(T_START_FRAC * len(pipeline.scheduler.timesteps))
        print(f"Using {NUM_INFERENCE_STEPS} inference steps.")
        print(f"Starting denoising at step index: {t_start_index} (out of {len(pipeline.scheduler.timesteps)})")

        # Determine padding token value
        PADD_TOKEN = len(pipeline.gate_pool) + 1
        print(f"Gate pool size: {len(pipeline.gate_pool)}. Padding token index: {PADD_TOKEN}")
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        print("Cannot proceed with experiments. Set GENERATE_VISUALS_ONLY = True to only generate masks.")
        # Exit or raise error if model is essential for the intended run
        if not GENERATE_VISUALS_ONLY: raise # Stop if experiments were requested but model failed

# Create output directories if they don't exist
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved in: {RESULTS_OUTPUT_DIR}")
print(f"Circuit/Mask images will be saved in: {IMAGE_OUTPUT_DIR}")


# %% Main Experiment Loop

results = []
start_time = time.time()

for num_qubits in QUBIT_COUNTS:
    print(f"\n===== Processing {num_qubits} Qubits =====")
    system_size = num_qubits # Assuming model trained for exact qubit count

    # 1. Get Geometries (including Baseline)
    if num_qubits not in HARDWARE_GEOMETRIES:
        print(f"No defined geometries for {num_qubits} qubits. Skipping.")
        continue
    # Ensure Baseline is processed first for clarity/comparison if needed
    geometries_to_process = collections.OrderedDict()
    if "Baseline_Unconstrained" in HARDWARE_GEOMETRIES[num_qubits]:
        geometries_to_process["Baseline_Unconstrained"] = HARDWARE_GEOMETRIES[num_qubits]["Baseline_Unconstrained"]
    for name, conn in HARDWARE_GEOMETRIES[num_qubits].items():
        if name != "Baseline_Unconstrained":
            geometries_to_process[name] = conn

    # 2. Get SRVs (Only if running experiments)
    target_srvs = None
    if not GENERATE_VISUALS_ONLY:
        target_srvs = generate_srvs(num_qubits)
        print(f"Found {len(target_srvs)} unique, valid SRVs for {num_qubits} qubits.")
        if not target_srvs:
            print(f"No valid SRVs found for {num_qubits} qubits after filtering. Skipping experiments for this qubit count.")
            # Continue to next qubit count if experiments were intended but no SRVs found
            continue

    # 3. Iterate through geometries
    for geo_name, connectivity in geometries_to_process.items(): # Use ordered dict
        print(f"\n--- Geometry: {geo_name} ---")
        sanitized_geo_name = sanitize_filename(geo_name) # Sanitize once per geometry

        # 3.1 Create the mask (passing geo_name)
        mask, effective_gates = create_mask_from_connectivity(
            num_qubits=num_qubits,
            geo_name=geo_name, # Pass name
            connectivity=connectivity,
            max_gates=MAX_GATES,
            gate_slot_duration=GATE_SLOT_DURATION
        )

        # Check if mask is effectively empty AFTER creation (should only happen for non-baseline)
        if effective_gates <= 0 and geo_name != "Baseline_Unconstrained":
             print(f"Skipping geometry {geo_name} because the generated mask has zero effective gates.")
             continue
        print(f"Mask created with effective gate sequence length: {effective_gates}")

        # --- Define Image Directory ---
        # Path structure: images/<qubits>/<geometry>/
        img_dir = os.path.join(IMAGE_OUTPUT_DIR, str(num_qubits), sanitized_geo_name)
        os.makedirs(img_dir, exist_ok=True)

        # --- Save Mask Image ---
        mask_image_path = os.path.join(img_dir, "mask.png")
        fig_mask = None # Initialize
        try:
            fig_mask, ax_mask = plt.subplots(figsize=(max(5, MAX_GATES/5), max(3, num_qubits/1.5)))
            cmap = matplotlib.colors.ListedColormap(['white', 'green'])
            bounds = [-0.5, 0.5, 1.5]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            im = ax_mask.imshow(mask.cpu().numpy(), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
            ax_mask.set_title(f"Mask: {geo_name} ({num_qubits}Q)\nEffective Gates: {effective_gates}", fontsize=10)
            ax_mask.set_xlabel("Gate Sequence Step", fontsize=9)
            ax_mask.set_ylabel("Qubit Index", fontsize=9)
            ax_mask.set_xticks(np.arange(0, MAX_GATES, step=max(1, MAX_GATES // 10)))
            ax_mask.set_yticks(np.arange(num_qubits))
            ax_mask.tick_params(axis='both', which='major', labelsize=8)
            ax_mask.set_xticks(np.arange(-.5, MAX_GATES, 1), minor=True)
            ax_mask.set_yticks(np.arange(-.5, num_qubits, 1), minor=True)
            ax_mask.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)
            fig_mask.savefig(mask_image_path, bbox_inches='tight', dpi=100)
            print(f"Saved mask image to: {mask_image_path}")
        except Exception as mask_err:
            print(f"\nError saving mask image for {num_qubits}Q, {geo_name}: {mask_err}")
        finally:
            if fig_mask:
                plt.close(fig_mask) # Close the mask figure
        # --- End Save Mask Image ---

        # --- Save Connectivity Graph Image (Skip for Baseline) ---
        if geo_name != "Baseline_Unconstrained" and connectivity: # Check if connectivity exists
            graph_image_path = os.path.join(img_dir, "graph.png")
            fig_graph = None # Initialize
            try:
                G = nx.Graph()
                G.add_nodes_from(range(num_qubits))
                G.add_edges_from(connectivity)

                fig_graph, ax_graph = plt.subplots(figsize=(6, 6))
                # Use a layout that spreads nodes well
                pos = nx.kamada_kawai_layout(G)
                nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color='lightblue',
                        node_size=500, font_size=10, edge_color='gray')
                ax_graph.set_title(f"Connectivity: {geo_name} ({num_qubits}Q)", fontsize=12)
                plt.tight_layout()
                fig_graph.savefig(graph_image_path, dpi=100)
                print(f"Saved graph image to: {graph_image_path}")
            except Exception as graph_err:
                print(f"\nError saving graph image for {num_qubits}Q, {geo_name}: {graph_err}")
            finally:
                if fig_graph:
                    plt.close(fig_graph) # Close the graph figure
        # --- End Save Connectivity Graph Image ---


        # === Conditionally Skip Experiments ===
        if GENERATE_VISUALS_ONLY:
            print(f"Skipping experiments for {geo_name} as GENERATE_VISUALS_ONLY is True.")
            # Clean up mask tensor even if skipping experiments
            if 'mask' in locals() and mask is not None:
                del mask
                util.MemoryCleaner.purge_mem()
            continue # Move to the next geometry
        # =====================================

        # Check if pipeline loaded successfully before proceeding
        if pipeline is None:
             print("Diffusion pipeline not loaded. Skipping experiments.")
             continue

        # 4. Iterate through target SRVs (Only if not GENERATE_VISUALS_ONLY)
        progress_bar = tqdm(target_srvs, desc=f"SRVs ({geo_name} - {num_qubits}Q)", leave=False)
        for target_srv in progress_bar:
            progress_bar.set_postfix({"SRV": str(target_srv)})

            # Calculate SRV complexity for path
            srv_complexity = str(target_srv).count('2')
            sanitized_target_srv_str = sanitize_filename(str(target_srv))

            # 4.1 Prepare the prompt
            prompt = f"Generate SRV: {target_srv}"

            # 4.2 Run the generation N times
            generated_qc = None # Initialize in case of error
            generated_srvs = None
            errors = -1
            run_failed = False # Flag for this specific run
            try:
                 generated_qc, generated_srvs, errors = generate_circuits_with_mask(
                    pipeline=pipeline,
                    prompt=prompt,
                    mask=mask, # Use the correctly generated mask
                    samples=SAMPLES_PER_EXPERIMENT,
                    system_size=system_size,
                    num_of_qubits=num_qubits,
                    max_gates=MAX_GATES,
                    target_num_gates=effective_gates, # Use effective gates from mask
                    target_num_bits=num_qubits,
                    guidance_scale=GUIDANCE_SCALE,
                    t_start_index=t_start_index,
                    no_bar=True, # Hide inner progress bar
                    padd_tok_val=PADD_TOKEN
                )
            except torch.cuda.OutOfMemoryError:
                 print(f"\nCUDA OutOfMemoryError occurred for {num_qubits}Q, {geo_name}, SRV {target_srv}. Skipping this configuration.")
                 util.MemoryCleaner.purge_mem() # Attempt to clear memory
                 correct_count_total = 0 # Set correct count to 0 for failed run
                 run_failed = True
                 # Continue to store result below, skipping image saving
            except Exception as e:
                print(f"\nAn unexpected error occurred during generation for {num_qubits}Q, {geo_name}, SRV {target_srv}: {e}")
                util.MemoryCleaner.purge_mem()
                correct_count_total = 0 # Set correct count to 0 for failed run
                run_failed = True
                # Continue to store result below, skipping image saving


            # 4.3 Count successful generations & Save Circuit Images (only if run didn't fail)
            correct_count_total = 0 # Total correct circuits found
            if not run_failed and generated_qc is not None and generated_srvs is not None:
                num_generated = len(generated_qc)

                # --- Separate correct and incorrect circuits ---
                correct_circuits_to_save = []
                incorrect_circuits_to_save = []
                for qc, gen_srv in zip(generated_qc, generated_srvs):
                    is_correct = (list(gen_srv) == list(target_srv))
                    if is_correct:
                        correct_count_total += 1 # Count all correct circuits
                        correct_circuits_to_save.append((qc, gen_srv, "Correct"))
                    else:
                        incorrect_circuits_to_save.append((qc, gen_srv, "Incorrect"))

                # --- Save Circuit Images (Prioritizing Correct) ---
                circuit_img_dir = os.path.join(img_dir, f"srv_comp_{srv_complexity}") # Use main img_dir
                os.makedirs(circuit_img_dir, exist_ok=True)
                saved_count = 0

                # Save correct circuits first
                circuits_to_plot = correct_circuits_to_save + incorrect_circuits_to_save
                plot_idx = 0

                for qc, gen_srv, correctness_label in circuits_to_plot:
                    if saved_count >= CIRCUITS_TO_SAVE:
                        break # Stop if we have saved enough images

                    # Create figure and axes
                    fig_qc = None # Initialize
                    try:
                        # Determine figsize based on effective gates for this circuit
                        current_fig_width_gates = MAX_GATES if geo_name == "Baseline_Unconstrained" else effective_gates
                        fig_qc, ax = plt.subplots(figsize=(max(6, current_fig_width_gates * 0.4), max(3, num_qubits * 0.5)))

                        # Draw the circuit
                        qc.draw("mpl", plot_barriers=False, ax=ax, fold=-1) # fold=-1 prevents line wrapping

                        # Add title
                        title = f"Target: {target_srv} | Gen: {gen_srv} ({correctness_label})\n{num_qubits}Q | Geo: {geo_name} | Eff. Gates: {effective_gates}"
                        ax.set_title(title, fontsize=9) # Smaller font size

                        # Define filename
                        filename = f"q{num_qubits}_srv_{sanitized_target_srv_str}_idx{plot_idx}_{correctness_label}.png"
                        image_path = os.path.join(circuit_img_dir, filename)

                        # Save the figure
                        fig_qc.savefig(image_path, bbox_inches='tight', dpi=100) # Adjust dpi if needed
                        saved_count += 1
                        plot_idx += 1

                    except Exception as draw_err:
                        print(f"\nError drawing/saving circuit {plot_idx} for {num_qubits}Q, {geo_name}, SRV {target_srv}: {draw_err}")
                    finally:
                        if fig_qc:
                           plt.close(fig_qc) # IMPORTANT: Close the figure to free memory
                # --- End Save Circuit Images ---

            # 4.4 Store results
            results.append({
                "Qubits": num_qubits,
                "Geometry": geo_name,
                "Connectivity": str(connectivity), # Store as string
                "TargetSRV": str(target_srv), # Store as string
                "Samples": SAMPLES_PER_EXPERIMENT,
                "UniqueGenerated": len(generated_qc) if generated_qc is not None else 0,
                "CorrectSRVCount": correct_count_total, # Store the total count found
                "RunFailed": run_failed, # Use the flag
                "MaskEffectiveGates": effective_gates,
                "ConversionErrors": errors if not run_failed else -1 # Mark conversion errors as -1 if run failed
            })

        # Clean up mask tensor from GPU after finishing a geometry
        # Mask might not exist if only generating masks and loop was skipped
        if 'mask' in locals() and mask is not None:
            del mask
            util.MemoryCleaner.purge_mem()


# %% Process and Display Results (Only if experiments were run)

if not GENERATE_VISUALS_ONLY:
    end_time = time.time()
    print(f"\n===== Experiment Complete =====")
    print(f"Total execution time (including setup): {end_time - start_time:.2f} seconds")

    if not results:
        print("No results generated (potentially all runs failed or were skipped).")
    else:
        # Convert results to Pandas DataFrame
        results_df = pd.DataFrame(results)

        # Calculate success rate (Correct SRV Count / Samples) - exclude failed runs from average
        # Create a temporary df excluding failed runs for rate calculation
        valid_runs_df = results_df[~results_df['RunFailed']].copy()
        if not valid_runs_df.empty:
            # Avoid division by zero if Samples is 0 for some reason
            valid_runs_df['SuccessRate'] = np.where(valid_runs_df['Samples'] > 0,
                                                    valid_runs_df['CorrectSRVCount'] / valid_runs_df['Samples'],
                                                    0)
        else:
            # Handle case where all runs failed
            results_df['SuccessRate'] = 0.0
            # Create empty valid_runs_df with columns for consistency if needed later
            valid_runs_df = pd.DataFrame(columns=results_df.columns.tolist() + ['SuccessRate'])


        # Calculate SRV Complexity
        results_df['SRV_Complexity'] = results_df['TargetSRV'].astype(str).apply(lambda s: s.count('2'))
        if not valid_runs_df.empty:
           valid_runs_df['SRV_Complexity'] = valid_runs_df['TargetSRV'].astype(str).apply(lambda s: s.count('2'))


        # Calculate Gate Bins
        gate_bins_calculated = False
        try:
          # Ensure MaskEffectiveGates is numeric before binning
          results_df['MaskEffectiveGates'] = pd.to_numeric(results_df['MaskEffectiveGates'], errors='coerce')
          # Keep rows with valid MaskEffectiveGates in results_df for potential reporting
          # results_df.dropna(subset=['MaskEffectiveGates'], inplace=True) # Maybe don't drop here, handle in grouping

          if not valid_runs_df.empty:
              valid_runs_df['MaskEffectiveGates'] = pd.to_numeric(valid_runs_df['MaskEffectiveGates'], errors='coerce')
              # Only bin on rows with valid numeric MaskEffectiveGates
              valid_mask_runs_df = valid_runs_df.dropna(subset=['MaskEffectiveGates']).copy()

              if not valid_mask_runs_df.empty and valid_mask_runs_df['MaskEffectiveGates'].nunique() > 1:
                  # Use pd.cut for potentially more intuitive bins based on ranges
                  # Adjust bins if baseline (MAX_GATES) dominates
                  num_bins = 5
                  try:
                      valid_mask_runs_df['GateBins'] = pd.cut(valid_mask_runs_df['MaskEffectiveGates'], bins=num_bins, duplicates='drop', right=False) # Use cut, left-inclusive bins
                  except ValueError: # If cut fails with 5 bins (e.g., few unique values), try fewer
                      print("Warning: Could not create 5 bins for GateBins, trying fewer.")
                      num_bins = max(1, valid_mask_runs_df['MaskEffectiveGates'].nunique() // 2) # Heuristic
                      valid_mask_runs_df['GateBins'] = pd.cut(valid_mask_runs_df['MaskEffectiveGates'], bins=num_bins, duplicates='drop', right=False)

                  # Merge the bins back into valid_runs_df
                  valid_runs_df = valid_runs_df.merge(valid_mask_runs_df[['GateBins']], left_index=True, right_index=True, how='left')
                  # Convert Interval to string and handle NaNs
                  valid_runs_df['GateBins'] = valid_runs_df['GateBins'].astype(str).fillna('N/A')
                  gate_bins_calculated = True
              else:
                  valid_runs_df['GateBins'] = 'N/A'
                  print("Not enough unique gate values for binning or only failed runs exist.")
          else: # valid_runs_df is empty
              results_df['GateBins'] = 'N/A' # Add column to main df if needed

        except Exception as e: # Catch errors during binning
           print(f"Error during Gate Bins calculation: {e}. Skipping GateBins analysis.")
           if not valid_runs_df.empty: valid_runs_df['GateBins'] = 'N/A' # Assign a placeholder
           results_df['GateBins'] = 'N/A'


        # --- Display Statistics (Using valid_runs_df for averages) ---
        print("\n--- Overall Statistics (All Runs - First 5 Rows) ---")
        print(results_df.head())
        print(f"\nTotal Runs: {len(results_df)}, Failed Runs: {results_df['RunFailed'].sum()}")

        if not valid_runs_df.empty:
            print("\n--- Average Success Rate per Qubit Count (Valid Runs Only) ---")
            avg_by_qubits = valid_runs_df.groupby('Qubits')['SuccessRate'].mean().reset_index()
            print(avg_by_qubits)

            print("\n--- Average Success Rate per Qubit Count and Geometry Type (Valid Runs Only) ---")
            avg_by_qubit_geo = valid_runs_df.groupby(['Qubits', 'Geometry'])['SuccessRate'].mean().reset_index()
            # Sort to potentially put Baseline first/last
            avg_by_qubit_geo['IsBaseline'] = avg_by_qubit_geo['Geometry'] == 'Baseline_Unconstrained'
            avg_by_qubit_geo = avg_by_qubit_geo.sort_values(by=['Qubits', 'IsBaseline', 'Geometry']).drop(columns=['IsBaseline'])
            print(avg_by_qubit_geo)


            print("\n--- Average Success Rate per Qubit Count and SRV Complexity (Valid Runs Only) ---")
            avg_by_qubit_srv_complex = valid_runs_df[valid_runs_df['SRV_Complexity'] != 1].groupby(['Qubits', 'SRV_Complexity'])['SuccessRate'].mean().reset_index()
            print(avg_by_qubit_srv_complex)

            print("\n--- Average Success Rate per Qubit Count and Mask Effective Gates (Binned, Valid Runs Only) ---")
            if gate_bins_calculated and 'GateBins' in valid_runs_df.columns and valid_runs_df['GateBins'].nunique() > 1:
                 # Exclude 'N/A' category from average calculation if present
                 avg_by_qubit_gate_bins = valid_runs_df[valid_runs_df['GateBins'] != 'N/A'].groupby(['Qubits', 'GateBins'])['SuccessRate'].mean().reset_index()
                 print(avg_by_qubit_gate_bins)
            else:
                print("Skipping GateBins summary due to previous errors or lack of data in valid runs.")

            print("\n--- Best Performing Configurations (Valid Runs, Top 5 by Success Rate) ---")
            print(valid_runs_df.nlargest(5, 'SuccessRate')[["Qubits", "Geometry", "TargetSRV", "SuccessRate"]])

            print("\n--- Worst Performing Configurations (Valid Runs, Bottom 5 by Success Rate) ---")
            print(valid_runs_df.nsmallest(5, 'SuccessRate')[["Qubits", "Geometry", "TargetSRV", "SuccessRate"]])
        else:
            print("\nNo valid runs completed successfully. Cannot calculate average success rates.")


        # --- Save Results to CSV (Save the complete results_df) ---
        detailed_filename = os.path.join(RESULTS_OUTPUT_DIR, f"diffusion_mask_analysis_detailed_{timestamp}.csv")
        # Save summaries based on valid runs if available
        summary_qubits_filename = os.path.join(RESULTS_OUTPUT_DIR, f"summary_by_qubits_{timestamp}.csv")
        summary_qubit_geo_filename = os.path.join(RESULTS_OUTPUT_DIR, f"summary_by_qubit_geometry_{timestamp}.csv")
        summary_qubit_srv_filename = os.path.join(RESULTS_OUTPUT_DIR, f"summary_by_qubit_srv_complexity_{timestamp}.csv")
        summary_qubit_gates_filename = os.path.join(RESULTS_OUTPUT_DIR, f"summary_by_qubit_gate_bins_{timestamp}.csv")

        try:
            print(f"\nSaving detailed results (all runs) to: {detailed_filename}")
            results_df.to_csv(detailed_filename, index=False) # Save all data including failures

            if not valid_runs_df.empty:
                print(f"Saving summary by qubits (valid runs) to: {summary_qubits_filename}")
                avg_by_qubits.to_csv(summary_qubits_filename, index=False)

                print(f"Saving summary by qubit and geometry (valid runs) to: {summary_qubit_geo_filename}")
                avg_by_qubit_geo.to_csv(summary_qubit_geo_filename, index=False)

                print(f"Saving summary by qubit and SRV complexity (valid runs) to: {summary_qubit_srv_filename}")
                avg_by_qubit_srv_complex.to_csv(summary_qubit_srv_filename, index=False)

                if gate_bins_calculated and 'GateBins' in valid_runs_df.columns and valid_runs_df['GateBins'].nunique() > 1:
                    print(f"Saving summary by qubit and gate bins (valid runs) to: {summary_qubit_gates_filename}")
                    avg_by_qubit_gate_bins.to_csv(summary_qubit_gates_filename, index=False)
            else:
                print("Skipping saving of summary CSVs as no valid runs were completed.")

            print("CSV files saved successfully.")

        except Exception as e:
            print(f"\nError saving results to CSV: {e}")

else: # GENERATE_VISUALS_ONLY is True
    end_time = time.time()
    print(f"\n===== Mask & Graph Generation Complete =====") # Updated message
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("Skipped experiments and results processing as GENERATE_VISUALS_ONLY was set to True.")


# %% Cleanup
pipeline = None # Release pipeline object
util.MemoryCleaner.purge_mem()
print("\nScript finished.")
