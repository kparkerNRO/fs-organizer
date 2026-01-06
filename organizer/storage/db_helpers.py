from sqlalchemy import (
    TypeDecorator,
    String,
)
import json
from datetime import datetime


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


class DateTime(TypeDecorator):
    """Custom type for handling datetime objects stored as ISO format strings"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert datetime to ISO format string for storage"""
        if value is not None:
            if isinstance(value, datetime):
                return value.isoformat()
            # If already a string, pass it through (for legacy compatibility)
            return value
        return None

    def process_result_value(self, value, dialect):
        """Convert ISO format string back to datetime object"""
        if value is not None:
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            # If already a datetime, pass it through
            return value
        return None
