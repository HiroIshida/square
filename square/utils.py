import base64
import json
import pickle
from dataclasses import dataclass, fields
from typing import Any, Type, TypeVar, get_args, get_origin

import numpy as np

JsonableT = TypeVar("JsonableT", bound="Jsonable")


@dataclass
class Jsonable:
    @staticmethod
    def _serialize(value: Any, ttype: Type) -> str:
        if isinstance(value, Jsonable):
            return value.serialize()  # type: ignore
        elif isinstance(value, np.ndarray):
            return base64.b64encode(pickle.dumps(value)).decode("utf-8")
        elif isinstance(value, (int, str, float, bool)):
            return str(value)
        elif get_origin(ttype) in [list, tuple]:
            sub_types = get_args(ttype)
            assert len(sub_types) == 1
            sub_type = sub_types[0]
            return json.dumps([Jsonable._serialize(e, sub_type) for e in value])
        else:
            message = "type {} is not supported".format(ttype)
            assert False, message

    def serialize(self) -> str:
        d = {}
        for field in fields(self):
            key = field.name
            value = self.__dict__[key]
            ttype = field.type
            if isinstance(value, Jsonable):
                d[key] = value.serialize()  # type: ignore
            elif isinstance(value, np.ndarray):
                d[key] = base64.b64encode(pickle.dumps(value)).decode("utf-8")
            elif isinstance(value, (int, str, float, bool)):
                d[key] = value  # type: ignore
            elif get_origin(ttype) in [list, tuple]:
                pass
            else:
                message = "key {} of type {} is not supported".format(key, field.type)
                assert False, message
        return json.dumps(d)

    @classmethod
    def deserialize(cls: Type[JsonableT], string: str) -> JsonableT:
        d = json.loads(string)
        kwargs = {}
        for field in fields(cls):
            key = field.name
            ttype = field.type
            str_value = d[key]

            if issubclass(ttype, Jsonable):
                kwargs[key] = ttype.deserialize(str_value)
            elif issubclass(ttype, np.ndarray):
                kwargs[key] = pickle.loads(base64.b64decode(str_value.encode()))
            elif ttype in [int, str, float, bool]:
                kwargs[key] = ttype(d[key])
            else:
                assert False
        return cls(**kwargs)  # type: ignore
