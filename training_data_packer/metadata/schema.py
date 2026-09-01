import json
from importlib import resources

import jsonschema
from referencing import Registry, Resource

from training_data_packer.metadata import Metadata

_resource_dir = resources.files("training_data_packer.metadata").joinpath("resources")


def _load_json_resource(name: str) -> dict:
    text = _resource_dir.joinpath(name).read_text()
    return json.loads(text)


class Validator:
    def __init__(self, schema_name):
        self.registry = Registry()
        for s in schema_name:
            schema = _load_json_resource(s)
            self.registry = self.registry.with_resource(schema["$id"], Resource.from_contents(schema))

    def _validator(self, schema_url: str, data: dict) -> tuple[bool, str | None]:
        schema = self.registry.resolver().lookup(schema_url).contents
        validator = jsonschema.Draft202012Validator(schema, registry=self.registry)
        try:
            validator.validate(data)
        except jsonschema.ValidationError as e:
            return False, f"Validation error for schema {schema_url}: {e}"
        return True, None

    def validate_metadata(self, metadata: Metadata) -> tuple[bool, str | None]:
        return self._validator("https://https://openeurollm.eu/schemas/metadata.json", dict(metadata))

    def validate_release_part(self, data: dict) -> tuple[bool, str | None]:
        return self._validator("https://https://openeurollm.eu/schemas/part.json", data)
