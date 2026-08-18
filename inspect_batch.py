import hydra
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig

with initialize_config_module("perturbench.configs", version_base=None):
    cfg = compose(
        config_name="train",
        overrides=["experiment=neurips2025/norman19/linear_best_params_norman19"],
        return_hydra_config=True,
    )
    HydraConfig.instance().set_config(cfg)

dm = hydra.utils.instantiate(cfg.data)
dm.setup(stage="fit")          # ← 여기만 수정

batch = next(iter(dm.train_dataloader()))

print("batch type:", type(batch))
print("gene_expression:", batch.gene_expression.shape, batch.gene_expression.dtype)
print("perturbations :", batch.perturbations.shape, batch.perturbations.dtype)
print("  sample[0]   :", batch.perturbations[0])
for k, v in batch.covariates.items():
    print(f"covariate '{k}':", v.shape, v.dtype)
    print("  sample[0] :", v[0])