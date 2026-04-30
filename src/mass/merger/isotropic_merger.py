import copy
import logging
from mass.merger.merger import TaskVectorBasedMerger
from mass.modules.encoder import ImageEncoder
from mass.utils.utils import apply_dict_to_model, compute_task_dict, print_memory, sum_task_dict
from mass.utils.task_vectors import isotropic_sum

import torch

pylogger = logging.getLogger(__name__)


class IsotropicMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, model_name, device="cuda", **kwargs):
        super().__init__()

        self.optimal_alphas = optimal_alphas
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        pylogger.info(f"IsotropicMerger initialized on device: {self.device}")

    @torch.no_grad()
    def merge(self, base_model, finetuned_models):

        base_model = base_model.to(self.device)
        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets)

        cumulative_dict = {}
        for dataset in datasets:
            ft_state_dict = {k: v.to(self.device) for k, v in finetuned_models[dataset].items()}
            task_dict = compute_task_dict(base_model.state_dict(), ft_state_dict)
            cumulative_dict = sum_task_dict(cumulative_dict, task_dict)
            del finetuned_models[dataset]
            del ft_state_dict
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        multi_task_vector = isotropic_sum(cumulative_dict, datasets, device=str(self.device))

        coefficient = 1.0
        if (
            self.model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[self.model_name]
        ):
            coefficient = self.optimal_alphas[self.model_name][num_tasks]

        pylogger.info(f"IsotropicMerger using alpha={coefficient} for model={self.model_name}, tasks={num_tasks}")

        merged_encoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(multi_task_vector, merged_encoder, coefficient=coefficient)

        return merged_encoder


class IsotropicCommonTaskSpecificMerger(TaskVectorBasedMerger):
    def __init__(
        self,
        common_space_fraction,
        optimal_alphas,
        model_name,
        device,
    ):
        super().__init__()

        self.common_space_fraction = common_space_fraction
        self.optimal_alphas = optimal_alphas
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def merge(self, base_model, finetuned_models) -> ImageEncoder | None:

        multi_task_vector = {}
        task_dicts = {}

        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets)

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        pylogger.info("Computing SVD...")
        ref_task_dict = task_dicts[datasets[0]]
        for key in ref_task_dict:
            shape_ = ref_task_dict[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
            if not is_2d_matrix:
                pylogger.info(f"Combining by avg {key}...")

                for i, (dataset, task_dict) in enumerate(task_dicts.items()):
                    vec = task_dict[key].to(self.device)
                    if i == 0:
                        multi_task_vector[key] = vec.clone()
                    else:
                        multi_task_vector[key] += (vec - multi_task_vector[key]) / (
                            i + 1
                        )
                continue

            pylogger.info(f"Computing common space using sum for {key}...")
            combined_w = sum(
                [task_dict[key].to(self.device) for task_dict in task_dicts.values()]
            )

            ### Calculate the common space size (making sure that task specific space is equally divisible) ###
            common_space_index_s = int(min(shape_) * self.common_space_fraction)
            _task_specific_total_space_index_s = (
                round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
            )
            common_space_index_s = min(shape_) - _task_specific_total_space_index_s

            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]
            common_space_s = s[:common_space_index_s]
            common_space_v = v[:common_space_index_s, :]
            ###################################################################

            ### Calculate task specific space ###
            n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
            for i, task_dict in enumerate(task_dicts.values()):
                w = task_dict[key].to(self.device)

                # calculate the projection onto task specific space to remove the common space
                w_ts = w - common_space_u @ common_space_u.T @ w
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                if i == 0:
                    combined_space_u = torch.zeros_like(u_ts, device=self.device)
                    combined_space_s = torch.zeros_like(s_ts, device=self.device)
                    combined_space_v = torch.zeros_like(v_ts, device=self.device)

                combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    u_ts[:, :n_dims_per_task]
                )
                combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    s_ts[:n_dims_per_task]
                )
                combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = (
                    v_ts[:n_dims_per_task, :]
                )
            ###################################################################

            combined_space_u[
                :,
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
            ] = common_space_u
            combined_space_s[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s
            ] = common_space_s
            combined_space_v[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
                :,
            ] = common_space_v

            ### Orthogonalize combined_space_u and combined_space_v ###
            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u, full_matrices=False)
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v, full_matrices=False)
            )
            combined_space_u = u_combined_space_u @ v_combined_space_u
            combined_space_v = u_combined_space_v @ v_combined_space_v
            ###################################################################

            combined_space_s = (
                torch.ones_like(combined_space_s) * combined_space_s.mean()
            )

            multi_task_vector[key] = torch.linalg.multi_dot(
                (
                    combined_space_u,
                    torch.diag(combined_space_s),
                    combined_space_v,
                )
            )

        coefficient = self.optimal_alphas[self.model_name][num_tasks]

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
