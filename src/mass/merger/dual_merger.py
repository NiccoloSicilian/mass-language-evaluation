import copy
import gc
import logging

import torch

from mass.merger.merger import TaskVectorBasedMerger
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory
from mass.utils.task_vectors import avg_layers, get_svd_dict, sum_svd
from mass.utils.dual_arithmetic import build_duality_map, get_t5_topological_order, get_vit_topological_order

pylogger = logging.getLogger(__name__)


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
        task_dicts = {}
        for dataset in datasets:
            ft_state_dict = {
                k: v.to(self.device) for k, v in finetuned_models[dataset].items()
            }
            task_dicts[dataset] = compute_task_dict(base_model.state_dict(), ft_state_dict)
            del ft_state_dict
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        print_memory("after computing task dicts")

        # ── Step 2: SVD decomposition ─────────────────────────────────────────
        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

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

        pylogger.info(f"Ordered keys ({len(ordered_keys)}): {ordered_keys[:4]} ...")

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

        multi_task_vector_cpu = {
            k: v.to(self.device) for k, v in multi_task_vector_cpu.items()
        }

        del dualized
        gc.collect()

        # ── Step 7: apply to base model ───────────────────────────────────────
        coefficient = 1.0
        if (
            self.model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[self.model_name]
        ):
            coefficient = self.optimal_alphas[self.model_name][num_tasks]

        pylogger.info(
            f"DualMerger alpha={coefficient}, model={self.model_name}, tasks={num_tasks}"
        )

        merged_model = copy.deepcopy(base_model)
        merged_model = apply_dict_to_model(
            multi_task_vector_cpu, merged_model, coefficient=coefficient
        )

        return merged_model
