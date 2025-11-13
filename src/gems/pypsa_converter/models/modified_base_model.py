from pydantic import BaseModel


class ModifiedBaseModel(BaseModel):
    class Config:
        alias_generator = lambda snake: snake.replace("_", "-")
        extra = "forbid"
        populate_by_name = True
