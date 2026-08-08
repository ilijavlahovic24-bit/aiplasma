#Data processing includes:
#Data ingestion
#Data preprocessing
#Data exploration
#Feature engineering
#Data splitting
from typing import Optional

from data_parser import DataSource
from data_transformer import TransformerConfig, DataTransformer, FeatureBatchDataset


class DataPipeline:
    def __init__(self,sources: list[DataSource], config: Optional[TransformerConfig] = None):
        self.sources = sources
        self.config = config or TransformerConfig()
    def build(self):
        # ── Validacija pre učitavanja ────────────────────────────
        for source in self.sources:
            if not source.validate():
                raise ValueError(
                    f"DataSource validacija nije prošla: {source.describe()}"
                )

        # ── Učitavanje ───────────────────────────────────────────
        tensors = [source.load() for source in self.sources]

        # ── Transformacija → lista FeatureBatch objekata ─────────
        transformer = DataTransformer(config=self.config)
        batches = transformer.transform(tensors)

        # ── Split na train i val ─────────────────────────────────
        dataset = FeatureBatchDataset(batches)
        return dataset.split(val_ratio=self.config.val_ratio)

    def describe(self) -> str:
        return (
            f"DataPipeline(\n"
            f"  sources = {len(self.sources)},\n"
            f"  val_ratio = {self.config.val_ratio},\n"
            f"  boundary_threshold = {self.config.boundary_threshold}\n"
            f")"
        )