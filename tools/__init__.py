# tools/__init__.py
from tools.weather import WEATHER_TOOL, get_weather
from tools.document import DOCUMENT_TOOL, search_document

TOOLS = [
    WEATHER_TOOL,
    DOCUMENT_TOOL
]