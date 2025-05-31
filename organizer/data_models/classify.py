from enum import Enum


class ClassificationType(str, Enum):
    VARIANT = "variant"
    CATEGORY = "category"
    SUBJECT = "subject"
    UNKNOWN = "unknown"
    CLUSTERED = "clustered"
