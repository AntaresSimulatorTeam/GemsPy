from pydantic import BaseModel


class ModifiedBaseModel(BaseModel):
    @staticmethod
    def _to_kebab(snake: str) -> str:
        return snake.replace("_", "-")

    class Config:
        alias_generator = _to_kebab
        extra = "forbid"
        populate_by_name = True
