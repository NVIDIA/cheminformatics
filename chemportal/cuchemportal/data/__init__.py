
class BaseModel():
    """
    Base class for all models.
    """

    @classmethod
    def from_dict(cls, record: dict):
        obj = cls()
        for key, value in record.items():
            setattr(obj, key, value)
        return obj
