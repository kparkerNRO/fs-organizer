from sqlalchemy import (
    TypeDecorator,
    String,
)
import json


class JsonList(TypeDecorator):
    """Custom type for handling lists stored as JSON strings"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return []


class JsonDict(TypeDecorator):
    """Custom type for handling dictionaries stored as JSON strings"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return {}
