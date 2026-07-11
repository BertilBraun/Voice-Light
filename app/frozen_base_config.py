from __future__ import annotations

from pydantic import BaseModel, ConfigDict

frozen_base_config = ConfigDict(frozen=True)


class FrozenBaseModel(BaseModel):
    model_config = frozen_base_config
