import copy
import gc
import logging

import torch

from mass.merger.merger import TaskVectorBasedMerger
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory, sum_task_dict
from mass.utils.task_vectors import avg_layers, get_svd_dict, sum_svd
from mass.utils.dual_arithmetic import build_duality_map, get_t5_topological_order, get_vit_topological_order
import os
from pathlib import Path
pylogger = logging.getLogger(__name__)

def save_task_vectors(
    task_vectors_dict,
    save_dir,
    model_name,
    num_tasks,
    aggregation_mode,
    mass_schedule,
    datasets=None
):
    """
    Save the processed multi-task vector dictionary to disk.
    
    Args:
        task_vectors_dict: Dict {layer_name -> tensor} of task vectors
        save_dir: Directory to save the checkpoint
        model_name: Model identifier (e.g., "t5-base", "ViT-B-32")
        num_tasks: Number of tasks merged
        aggregation_mode: "avg" or "tsv"
        mass_schedule: "uniform" or "linear"
        datasets: Optional list of dataset names for filename
    
    Returns:
        Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct filename
    dataset_str = "_".join(datasets) if datasets else "merged"
    if len(dataset_str) > 100:  # Truncate if too long
        dataset_str = f"{len(datasets)}tasks"
    
    filename = (
        f"task_vectors_{model_name}_{dataset_str}_"
        f"n{num_tasks}_{aggregation_mode}_{mass_schedule}.pt"
    )
    # Sanitize filename (replace problematic characters)
    filename = filename.replace("/", "_").replace(" ", "_")
    
    save_path = Path(save_dir) / filename
    
    # Save with metadata
    checkpoint = {
        "task_vectors": task_vectors_dict,
        "metadata": {
            "model_name": model_name,
            "num_tasks": num_tasks,
            "aggregation_mode": aggregation_mode,
            "mass_schedule": mass_schedule,
            "datasets": datasets,
            "num_layers": len(task_vectors_dict),
        }
    }
    
    torch.save(checkpoint, save_path)
    pylogger.info(f"Saved task vectors to: {save_path}")
    
    return save_path
    
class DualMerger(TaskVectorBasedMerger):

    def __init__(
        self,
        optimal_alphas,
        svd_path,
        svd_compress_factor,
        model_name,
        aggregation_mode,
        mass_schedule,
        device=None,
    ):
        """
        Args:
            optimal_alphas:      Nested dict {model_name -> {num_tasks -> alpha}}.
            svd_path:            Path to cache the pre-computed SVD dict.
            svd_compress_factor: SVD compression factor (1.0 = no compression).
            model_name:          Model identifier (e.g. "t5-base", "ViT-B-32").
            aggregation_mode:    "avg" (arithmetic mean) or "tsv" (Task Singular Vectors).
            mass_schedule:       "uniform" or "linear".
            device:              Optional device override.
        """
        super().__init__()
        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregation_mode = aggregation_mode
        self.mass_schedule = mass_schedule
        pylogger.info(f"DualMerger initialised on device: {self.device}")

    @torch.no_grad()
    def merge(self, base_model, finetuned_models):
        base_model = base_model.to(self.device)

        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets)

        # ── Step 1: compute task vectors ─────────────────────────────────────
        cumulative_dict ={}
        for dataset in datasets:
            ft_state_dict = {k: v.to(self.device) for k, v in finetuned_models[dataset].items()}
            task_dict = compute_task_dict(base_model.state_dict(), ft_state_dict)
            cumulative_dict = sum_task_dict(cumulative_dict, task_dict)
            del finetuned_models[dataset]
            del ft_state_dict
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")
        '''
        # ── Step 3: aggregate task vectors ───────────────────────────────────
        if self.aggregation_mode == "avg":
            multi_task_vector = avg_layers(
                svd_dict=svd_dict,
                device=str(self.device),
            )
        elif self.aggregation_mode == "tsv":
            multi_task_vector = sum_svd(
                ref_state_dict=copy.deepcopy(base_model.state_dict()),
                svd_dicts=svd_dict,
                non_matrix_params_aggregation="mean",
                device=str(self.device),
            )
        else:
            pylogger.error(f"Unknown aggregation_mode: '{self.aggregation_mode}'")
            return None
        '''
        multi_task_vector = cumulative_dict
        # ── Step 4: move to CPU before dualisation ───────────────────────────
        multi_task_vector_cpu = {k: v.cpu() for k, v in multi_task_vector.items()}
        del multi_task_vector
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        # ── Step 5: topological ordering ─────────────────────────────────────
        raw_keys = list(multi_task_vector_cpu.keys())
        if "t5" in self.model_name.lower():
            ordered_keys = get_t5_topological_order(raw_keys)
        else:
            ordered_keys = get_vit_topological_order(raw_keys)

        print(f"Ordered keys ({len(ordered_keys)}): {ordered_keys[:5]} ...")

        # ── Step 6: apply duality map ─────────────────────────────────────────
        dualized = build_duality_map(
            ordered_keys,
            multi_task_vector_cpu,
            self.device,
            self.mass_schedule,
            self.model_name,
        )

        for key in dualized:
            multi_task_vector_cpu[key] = dualized[key]
        for key in multi_task_vector_cpu:
            if key not in dualized:
                multi_task_vector_cpu[key] = multi_task_vector_cpu[key]/ num_tasks
        save_task_vectors(
                task_vectors_dict=multi_task_vector_cpu,
                save_dir=self.task_vectors_save_dir,
                model_name=self.model_name,
                num_tasks=num_tasks,
                aggregation_mode=self.aggregation_mode,
                mass_schedule=self.mass_schedule,
                datasets=datasets,
            )
        multi_task_vector_cpu = {
            k: v.to(self.device) for k, v in multi_task_vector_cpu.items()
        }

        del dualized
        gc.collect()

        # ── Step 7: apply to base model ───────────────────────────────────────
        # OmegaConf stores YAML integer keys as strings, so check both int and str.
        coefficient = 1.0
        if self.model_name in self.optimal_alphas:
            alpha_map = self.optimal_alphas[self.model_name]
            for key in (num_tasks, str(num_tasks)):
                if key in alpha_map:
                    coefficient = float(alpha_map[key])
                    break

        print(
            f"DualMerger alpha={coefficient}, model={self.model_name}, tasks={num_tasks}"
        )

        merged_model = copy.deepcopy(base_model)
        merged_model = apply_dict_to_model(
            multi_task_vector_cpu, merged_model, coefficient=coefficient
        )

        return merged_model
